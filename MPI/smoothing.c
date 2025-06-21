#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

#define CHANNELS 3

int clamp(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return val;
}

// Used mean filter
void smooth_chunk(unsigned char* in, unsigned char* out, int width, int height, int start_row, int end_row) {
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < width; x++) {
            int sum[3] = {0, 0, 0}, count = 0;
            for (int ky = -1; ky <= 1; ky++) {
                int yy = y + ky;
                if (yy < 0 || yy >= height) continue;
                for (int kx = -1; kx <= 1; kx++) {
                    int xx = x + kx;
                    if (xx < 0 || xx >= width) continue;
                    int idx = (yy * width + xx) * CHANNELS;
                    sum[0] += in[idx];
                    sum[1] += in[idx + 1];
                    sum[2] += in[idx + 2];
                    count++;
                }
            }
            int idx = (y * width + x) * CHANNELS;
            out[idx]     = clamp(sum[0] / count);
            out[idx + 1] = clamp(sum[1] / count);
            out[idx + 2] = clamp(sum[2] / count);
        }
    }
}

int main(int argc, char** argv) {
    int rank, size, width, height, channels;
    unsigned char *img = NULL, *out = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s input_image output_image\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Root process loads the image
    if (rank == 0) {
        img = stbi_load(argv[1], &width, &height, &channels, CHANNELS);
        if (!img) {
            fprintf(stderr, "Error loading image %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        out = malloc(width * height * CHANNELS);
    }

      return 0;
}
