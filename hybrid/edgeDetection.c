#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

int main(int argc, char *argv[]) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s input_image output_image num_threads\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int num_threads = atoi(argv[3]);
    omp_set_num_threads(num_threads);

    char input_path[512], output_path[512];
    snprintf(input_path, sizeof(input_path), "../inputImages/%s", argv[1]);
    snprintf(output_path, sizeof(output_path), "../outputImages/%s", argv[2]);

    int width = 0, height = 0, channels = 0;
    unsigned char *img = NULL;

    if (rank == 0) {
        img = stbi_load(input_path, &width, &height, &channels, 3);
        if (!img) {
            fprintf(stderr, "Error: Could not load image %s\n", input_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int img_size = width * height * 3;
    if (rank != 0) img = malloc(img_size);
    MPI_Bcast(img, img_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int rows_per_proc = height / size;
    int extra = height % size;
    int start_row = rank * rows_per_proc + (rank < extra ? rank : extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);

    unsigned char *local_out = malloc(local_rows * width * 3);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    double start = MPI_Wtime();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < local_rows; y++) {
        for (int x = 0; x < width; x++) {
            int global_y = start_row + y;
            float edge_r_x = 0, edge_r_y = 0;
            float edge_g_x = 0, edge_g_y = 0;
            float edge_b_x = 0, edge_b_y = 0;

            for (int ky = -1; ky <= 1; ky++) {
                int yy = global_y + ky;
                if (yy < 0) yy = 0;
                if (yy >= height) yy = height - 1;
                for (int kx = -1; kx <= 1; kx++) {
                    int xx = x + kx;
                    if (xx < 0) xx = 0;
                    if (xx >= width) xx = width - 1;
                    int idx = (yy * width + xx) * 3;
                    int kernel_x = Gx[ky + 1][kx + 1];
                    int kernel_y = Gy[ky + 1][kx + 1];
                    edge_r_x += kernel_x * img[idx];
                    edge_r_y += kernel_y * img[idx];
                    edge_g_x += kernel_x * img[idx + 1];
                    edge_g_y += kernel_y * img[idx + 1];
                    edge_b_x += kernel_x * img[idx + 2];
                    edge_b_y += kernel_y * img[idx + 2];
                }
            }

            int mag_r = clamp((int)sqrt(edge_r_x * edge_r_x + edge_r_y * edge_r_y), 0, 255);
            int mag_g = clamp((int)sqrt(edge_g_x * edge_g_x + edge_g_y * edge_g_y), 0, 255);
            int mag_b = clamp((int)sqrt(edge_b_x * edge_b_x + edge_b_y * edge_b_y), 0, 255);

            int out_idx = (y * width + x) * 3;
            local_out[out_idx] = (unsigned char)mag_r;
            local_out[out_idx + 1] = (unsigned char)mag_g;
            local_out[out_idx + 2] = (unsigned char)mag_b;
        }
    }

    double end = MPI_Wtime();

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
        printf("Edge detection time: %.4f seconds\n", end - start);
        if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
            fprintf(stderr, "Error saving %s\n", output_path);
        } else {
            printf("Edge image saved to %s\n", output_path);
        }
        free(out);
        free(recvcounts);
        free(displs);
    }

    free(local_out);
    stbi_image_free(img);
    MPI_Finalize();
    return 0;
}
