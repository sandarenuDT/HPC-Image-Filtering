#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../stb_image.h"
#include "../stb_image_write.h"

// Clamp value to 0–255
int clamp(int val) {
    return (val < 0) ? 0 : (val > 255 ? 255 : val);
}
// double calculate_rmse(unsigned char *img1, unsigned char *img2, int size) {
//     double sum_sq_error = 0.0;
//     for (int i = 0; i < size; i++) {
//         double diff = (double)img1[i] - (double)img2[i];
//         sum_sq_error += diff * diff;
//     }
//     return sqrt(sum_sq_error / size);
// }

// 2D Gaussian function
float gaussian(float x, float y, float sigma) {
    float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
    return expf(exponent) / (2.0f * M_PI * sigma * sigma);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image> [sigma] [threads]\n", argv[0]);
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

    // Load input image
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return 1;
    }
    unsigned char *serial_img = stbi_load("../outputImages/serial_img.png", &width, &height, &channels, 3);
    if (!serial_img) {
        fprintf(stderr, "Error loading serial image for RMSE comparison\n");
        // free(out);
        stbi_image_free(img);
        return 1;
    }

    // Allocate memory for output
    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Failed to allocate memory for output image.\n");
        stbi_image_free(img);
        return 1;
    }

    // Gaussian kernel
    int radius = (int)ceil(3 * sigma);
    int kernel_size = 2 * radius + 1;
    float *kernel = malloc(kernel_size * kernel_size * sizeof(float));
    if (!kernel) {
        fprintf(stderr, "Failed to allocate memory for kernel.\n");
        free(out);
        stbi_image_free(img);
        return 1;
    }

    float sum = 0.0f;

    // kernel weights
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float w = gaussian((float)x, (float)y, sigma);
            kernel[(y + radius) * kernel_size + (x + radius)] = w;
            sum += w;
        }
    }

    // Normalizing
    #pragma omp parallel for
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] /= sum;
    }

    
    double start_time = omp_get_wtime();

    // Gaussian filter
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;

            // #pragma omp simd

            for (int ky = -radius; ky <= radius; ky++) {
                int yy = y + ky;
                yy = yy < 0 ? 0 : (yy >= height ? height - 1 : yy);

                for (int kx = -radius; kx <= radius; kx++) {
                    int xx = x + kx;
                    xx = xx < 0 ? 0 : (xx >= width ? width - 1 : xx);

                    float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
                    int idx = (yy * width + xx) * 3;

                    r += img[idx]     * weight;
                    g += img[idx + 1] * weight;
                    b += img[idx + 2] * weight;
                }
            }

            int out_idx = (y * width + x) * 3;
            out[out_idx]     = clamp((int)(r + 0.5f));
            out[out_idx + 1] = clamp((int)(g + 0.5f));
            out[out_idx + 2] = clamp((int)(b + 0.5f));
        }
    }

    double end_time = omp_get_wtime();
    printf("Smoothing completed (σ=%.2f) with %d threads in %.3f seconds.\n",
           sigma, num_threads, end_time - start_time);

    // Save result
    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Failed to save image to %s\n", output_path);
    }
    // int total_pixels = width * height * 3;
    // double rmse = calculate_rmse(out, serial_img, total_pixels);

    // printf("RMSE between parallel and serial images: %.4f\n", rmse);

    // double accuracy = (1.0 - (rmse / 255.0)) * 100.0;
    // if (accuracy < 0) accuracy = 0; // Clamp
    // printf("Accuracy of parallel code vs serial code: %.2f%%\n", accuracy);



    free(kernel);
    free(out);
    stbi_image_free(img);

    return 1;
}
