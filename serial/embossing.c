// embossing.c
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stb_image.h"
#include "stb_image_write.h"

// Clamp pixel values to [0, 255]
int clamp(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return val;
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

    clock_t start = clock();

    // Emboss filter: difference with upper-left neighbor + 128 offset
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;

            if (x == 0 || y == 0) {
                out[idx] = out[idx + 1] = out[idx + 2] = 128;
            } else {
                int ul_idx = ((y - 1) * width + (x - 1)) * 3;

                int diff_r = img[idx] - img[ul_idx];
                int diff_g = img[idx + 1] - img[ul_idx + 1];
                int diff_b = img[idx + 2] - img[ul_idx + 2];

                int max_diff = diff_r;
                if (abs(diff_g) > abs(max_diff)) max_diff = diff_g;
                if (abs(diff_b) > abs(max_diff)) max_diff = diff_b;

                int val = clamp(128 + max_diff);
                out[idx] = out[idx + 1] = out[idx + 2] = (unsigned char)val;
            }
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Embossing took %.4f seconds\n", elapsed);

    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Error saving image %s\n", output_path);
    } else {
        printf("Embossed image saved to %s\n", output_path);
    }

    free(out);
    stbi_image_free(img);
    return 0;
}
