#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

int clamp(int val) {
    return (val < 0) ? 0 : (val > 255) ? 255 : val;
}

// int main with optional thread count
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_image output_image [threads]\n", argv[0]);
        return 1;
    }

    // Optional thread count argument
    int num_threads = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);

    // Input/output paths
    char input_path[512];
    char output_path[512];
    snprintf(input_path, sizeof(input_path), "../inputImages/%s", argv[1]);
    snprintf(output_path, sizeof(output_path), "../outputImages/%s", argv[2]);

    // Load input image
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error loading %s\n", input_path);
        return 1;
    }

    // Allocate output buffer
    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    double start = omp_get_wtime();

    // Parallel embossing filter
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;

            if (x == 0 || y == 0) {
                out[idx] = out[idx+1] = out[idx+2] = 128; // Neutral gray
            } else {
                // upper-left neighbor
                int ul_idx = ((y-1) * width + (x-1)) * 3;

                // differences
                int diff_r = img[idx] - img[ul_idx];
                int diff_g = img[idx+1] - img[ul_idx+1];
                int diff_b = img[idx+2] - img[ul_idx+2];

                // absolute difference max
                int max_diff = diff_r;
                if (abs(diff_g) > abs(max_diff)) max_diff = diff_g;
                if (abs(diff_b) > abs(max_diff)) max_diff = diff_b;

                // emboss effect value
                int val = clamp(128 + max_diff);
                out[idx] = out[idx+1] = out[idx+2] = (unsigned char)val;
            }
        }
    }

    double end = omp_get_wtime();
    printf("Embossing completed with %d threads in %.4f seconds\n", num_threads, end - start);

    // Save output image
    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Error saving %s\n", output_path);
    } else {
        printf("Embossed image saved to %s\n", output_path);
    }

    free(out);
    stbi_image_free(img);

    return 0;
}
