// color_edge_openmp.c
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_image output_image [threads]\n", argv[0]);
        printf("Example: %s input.png edges.png 4\n", argv[0]);
        return 1;
    }

    // Set number of threads from argv or default to max
    int num_threads = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);

    char input_path[512];
    char output_path[512];
    snprintf(input_path, sizeof(input_path), "../inputImages/%s", argv[1]);
    snprintf(output_path, sizeof(output_path), "../outputImages/%s", argv[2]);

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error: Could not load image %s\n", input_path);
        return 1;
    }

    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Error: Could not allocate memory for output image\n");
        stbi_image_free(img);
        return 1;
    }

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

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float edge_r_x = 0, edge_r_y = 0;
            float edge_g_x = 0, edge_g_y = 0;
            float edge_b_x = 0, edge_b_y = 0;

            for (int ky = -1; ky <= 1; ky++) {
                int yy = y + ky;
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
            out[out_idx] = (unsigned char)mag_r;
            out[out_idx + 1] = (unsigned char)mag_g;
            out[out_idx + 2] = (unsigned char)mag_b;
        }
    }

    double end = omp_get_wtime();
    printf("Edge detection completed with %d threads in %.4f seconds\n", num_threads, end - start);

    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Error saving %s\n", output_path);
    } else {
        printf("Edge detected image saved to %s\n", output_path);
    }

    free(out);
    stbi_image_free(img);
    return 0;
}
