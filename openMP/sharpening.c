// sharpening_openmp.c
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

// Clamp pixel values between 0 and 255
int clamp(int val) {
    return (val < 0) ? 0 : (val > 255 ? 255 : val);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image> [threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    int num_threads = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();

    omp_set_num_threads(num_threads);

    char input_path[512], output_path[512];
    snprintf(input_path, sizeof(input_path), "../inputImages/%s", input_filename);
    snprintf(output_path, sizeof(output_path), "../outputImages/%s", output_filename);

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error loading image: %s\n", input_path);
        return 1;
    }

    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    // Sharpening kernel
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum_r = 0, sum_g = 0, sum_b = 0;

            for (int ky = -1; ky <= 1; ky++) {
                int yy = y + ky;
                if (yy < 0) yy = 0;
                if (yy >= height) yy = height - 1;

                for (int kx = -1; kx <= 1; kx++) {
                    int xx = x + kx;
                    if (xx < 0) xx = 0;
                    if (xx >= width) xx = width - 1;

                    int idx = (yy * width + xx) * 3;
                    int k = kernel[ky + 1][kx + 1];

                    sum_r += k * img[idx];
                    sum_g += k * img[idx + 1];
                    sum_b += k * img[idx + 2];
                }
            }

            int out_idx = (y * width + x) * 3;
            out[out_idx]     = clamp(sum_r);
            out[out_idx + 1] = clamp(sum_g);
            out[out_idx + 2] = clamp(sum_b);
        }
    }

    double end = omp_get_wtime();
    printf("Sharpening completed with %d threads in %.4f seconds\n", num_threads, end - start);

    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Error saving image %s\n", output_path);
    } else {
        printf("Sharpened image saved to %s\n", output_path);
    }

    free(out);
    stbi_image_free(img);

    return 0;
}
