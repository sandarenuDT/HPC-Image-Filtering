#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

// Clamp value to 0–255
int clamp(int val) {
    return (val < 0) ? 0 : (val > 255 ? 255 : val);
}

// 2D Gaussian function
float gaussian(float x, float y, float sigma) {
    float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
    return expf(exponent) / (2.0f * M_PI * sigma * sigma);
}

//  for boundary
int clamp_coord(int val, int max) {
    return (val < 0) ? 0 : (val > max) ? max : val;
}

int main(int argc, char *argv[]) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if(rank == 0)
            printf("Usage: %s <input_image> <output_image> [sigma] [threads]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    float sigma = (argc > 3) ? atof(argv[3]) : 0.85f;
    int num_threads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);

    char input_path[512], output_path[512];
    snprintf(input_path, sizeof(input_path), "../inputImages/%s", input_filename);
    snprintf(output_path, sizeof(output_path), "../outputImages/%s", output_filename);

    int width = 0, height = 0, channels = 0;
    unsigned char *img = NULL;

    // Only root loads the image
    if (rank == 0) {
        img = stbi_load(input_path, &width, &height, &channels, 3);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", input_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int img_size = width * height * 3;

    // Broadcast the image to all processes
    if (rank != 0) {
        img = malloc(img_size);
    }
    MPI_Bcast(img, img_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Gaussian kernel
    int radius = (int)ceil(3 * sigma);
    int kernel_size = 2 * radius + 1;
    float *kernel = malloc(kernel_size * kernel_size * sizeof(float));

    // Compute kernel
    float sum = 0.0f;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float w = gaussian((float)x, (float)y, sigma);
            kernel[(y + radius) * kernel_size + (x + radius)] = w;
            sum += w;
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; i++)
        kernel[i] /= sum;

    // Divide image rows among processes
    int rows_per_proc = height / size;
    int extra = height % size;
    int start_row = rank * rows_per_proc + (rank < extra ? rank : extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);

    // for local rows 
    int halo = radius;
    int local_rows_with_halo = local_rows + 2 * halo;
    unsigned char *local_in = malloc(local_rows_with_halo * width * 3);

    // Copy local data from global image to local buffer (with halo)
    for (int y = 0; y < local_rows_with_halo; y++) {
        int global_y = start_row + y - halo;
        int src_y = clamp_coord(global_y, height - 1);
        memcpy(&local_in[y * width * 3], &img[src_y * width * 3], width * 3);
    }

    // Free original image after distribution
    if (rank != 0) free(img);

    // Allocate output buffer for local rows (no halo)
    unsigned char *local_out = malloc(local_rows * width * 3);

    double start_time = MPI_Wtime();

    // OpenMP parallel Gaussian blur on local rows (skip halo rows)
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < local_rows; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int yy = y + ky + halo;
                for (int kx = -radius; kx <= radius; kx++) {
                    int xx = clamp_coord(x + kx, width - 1);
                    float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
                    int idx = (yy * width + xx) * 3;
                    r += local_in[idx]     * weight;
                    g += local_in[idx + 1] * weight;
                    b += local_in[idx + 2] * weight;
                }
            }
            int out_idx = (y * width + x) * 3;
            local_out[out_idx]     = clamp((int)(r + 0.5f));
            local_out[out_idx + 1] = clamp((int)(g + 0.5f));
            local_out[out_idx + 2] = clamp((int)(b + 0.5f));
        }
    }

    double end_time = MPI_Wtime();

    // Gather all processed rows to root
    int *recvcounts = NULL, *displs = NULL;
    unsigned char *out = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        out = malloc(img_size);

        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < extra ? 1 : 0);
            recvcounts[i] = rows * width * 3;
            displs[i] = (i * rows_per_proc + (i < extra ? i : extra)) * width * 3;
        }
    }

    MPI_Gatherv(local_out, local_rows * width * 3, MPI_UNSIGNED_CHAR,
                out, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Smoothing completed (σ=%.2f) with %d MPI processes and %d OpenMP threads per process in %.3f seconds.\n",
               sigma, size, num_threads, end_time - start_time);

        if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
            fprintf(stderr, "Failed to save image to %s\n", output_path);
        }
        free(out);
        free(recvcounts);
        free(displs);
    }

    free(kernel);
    free(local_in);
    free(local_out);

    MPI_Finalize();
    return 0;
}
