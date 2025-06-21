// edge_detection_serial.c
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../stb_image.h"
#include "../stb_image_write.h"


int clamp(int val) {
  return (val < 0) ? 0 : ((val > 255) ? 255 : val);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_image output_image\n", argv[0]);
        return 1;
    }

    char input_path[512], output_path[512];
    snprintf(input_path, sizeof(input_path), "../inputImages/%s", argv[1]);
    snprintf(output_path, sizeof(output_path), "../outputImages/%s", argv[2]);

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error loading image %s\n", input_path);
        return 1;
    }

    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    // Sobel kernels
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

    
    clock_t start = clock();

    //  Sobel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float edge_r_x = 0, edge_r_y = 0;
            float edge_g_x = 0, edge_g_y = 0;
            float edge_b_x = 0, edge_b_y = 0;

            // Apply 3x3 Sobel kernel with boundary clamping
            for (int ky = -1; ky <= 1; ky++) {
                int yy = y + ky;
                if (yy < 0) yy = 0;
                if (yy >= height) yy = height - 1;

                for (int kx = -1; kx <= 1; kx++) {
                    int xx = x + kx;
                    if (xx < 0) xx = 0;
                    if (xx >= width) xx = width - 1;

                    int idx = (yy * width + xx) * 3;
                    int kx_val = Gx[ky + 1][kx + 1];
                    int ky_val = Gy[ky + 1][kx + 1];

                    edge_r_x += kx_val * img[idx];
                    edge_r_y += ky_val * img[idx];
                    edge_g_x += kx_val * img[idx + 1];
                    edge_g_y += ky_val * img[idx + 1];
                    edge_b_x += kx_val * img[idx + 2];
                    edge_b_y += ky_val * img[idx + 2];
                }
            }

            int mag_r = clamp((int)sqrt(edge_r_x * edge_r_x + edge_r_y * edge_r_y));
            int mag_g = clamp((int)sqrt(edge_g_x * edge_g_x + edge_g_y * edge_g_y));
            int mag_b = clamp((int)sqrt(edge_b_x * edge_b_x + edge_b_y * edge_b_y));

            int out_idx = (y * width + x) * 3;
            out[out_idx] = (unsigned char)mag_r;
            out[out_idx + 1] = (unsigned char)mag_g;
            out[out_idx + 2] = (unsigned char)mag_b;
        }
    }

    
    clock_t end = clock();
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Edge detection took %.4f seconds\n", elapsed_secs);

    // Save 
    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Error saving image %s\n", output_path);
        free(out);
        stbi_image_free(img);
        return 1;
    }

    printf("Edge detected image saved to %s\n", output_path);

    free(out);
    stbi_image_free(img);
    return 0;
}
