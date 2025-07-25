#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "../stb_image.h"
#include "../stb_image_write.h"


int clamp(int val) {
    return (val < 0) ? 0 : (val > 255 ? 255 : val);
}

// 2D Gaussian function
float gaussian(float x, float y, float sigma) {
    float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
    return expf(exponent) / (2.0f * M_PI * sigma * sigma);
}

// Helper for boundary
int clamp_coord(int val, int max) {
    return (val < 0) ? 0 : (val > max) ? max : val;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if(rank == 0)
            printf("Usage: %s <input_image> <output_image> [sigma]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    float sigma = (argc > 3) ? atof(argv[3]) : 0.85f;

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

    // Compute kernel (all processes do this)
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

    // Allocate output buffer for local rows
    unsigned char *local_out = malloc(local_rows * width * 3);

    double start_time = MPI_Wtime();

    // Each process processes its portion
    for (int y = 0; y < local_rows; y++) {
        int global_y = start_row + y;
        for (int x = 0; x < width; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                int yy = clamp_coord(global_y + ky, height - 1);
                for (int kx = -radius; kx <= radius; kx++) {
                    int xx = clamp_coord(x + kx, width - 1);
                    float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
                    int idx = (yy * width + xx) * 3;
                    r += img[idx]     * weight;
                    g += img[idx + 1] * weight;
                    b += img[idx + 2] * weight;
                }
            }
            int out_idx = (y * width + x) * 3;
            local_out[out_idx]     = clamp((int)(r + 0.5f));
            local_out[out_idx + 1] = clamp((int)(g + 0.5f));
            local_out[out_idx + 2] = clamp((int)(b + 0.5f));
        }
    }

    double end_time = MPI_Wtime();

    // Gather all  to root
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
        printf("Smoothing completed (σ=%.2f) with %d processes in %.3f seconds.\n",
               sigma, size, end_time - start_time);

        if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
            fprintf(stderr, "Failed to save image to %s\n", output_path);
        }
        free(out);
        free(recvcounts);
        free(displs);
    }

    free(kernel);
    free(local_out);
    stbi_image_free(img);

    MPI_Finalize();
    return 0;
}
