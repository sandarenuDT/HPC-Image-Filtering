#include "../common/utils.h"
#include <math.h>


// 2D Gaussian function
float gaussian_2d(float x, float y, float sigma) {
    return expf(-(x*x + y*y)/(2*sigma*sigma)) / (2*M_PI*sigma*sigma);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_image output_image [sigma]\n", argv[0]);
        printf("Example: %s image.jpg blurred.png 1.2\n", argv[0]);
        return 1;
    }

    // Parameters with defaults
    float sigma = (argc > 3) ? atof(argv[3]) : 0.85f;  // Default σ = 0.85
    char input_path[512], output_path[512];
    build_paths(argv[1], argv[2], input_path, output_path);

    // Load image
    int width, height, channels;
    unsigned char *img = load_image(input_path, &width, &height);
    if (!img) {
        return 1;
    }

    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    // gaussian kernal generation
    int radius = (int)ceilf(3 * sigma);  // 3σ rule covers 99.7% of distribution
    int kernel_size = 2 * radius + 1;
    float *kernel = malloc(kernel_size * kernel_size * sizeof(float));
    float kernel_sum = 0.0f;

    // Generate kernel weights
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            float weight = gaussian_2d(kx, ky, sigma);
            kernel[(ky+radius)*kernel_size + (kx+radius)] = weight;
            kernel_sum += weight;
        }
    }

    // Normalizing kernel
    for (int i = 0; i < kernel_size*kernel_size; i++) {
        kernel[i] /= kernel_sum;
    }

    //filtering
    clock_t start = clock();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            // appplying kernel 
            for (int ky = -radius; ky <= radius; ky++) {
                int yy = y + ky;
                // boundary handling
                yy = (yy < 0) ? -yy : ((yy >= height) ? 2*height - yy - 2 : yy);

                for (int kx = -radius; kx <= radius; kx++) {
                    int xx = x + kx;
                    xx = (xx < 0) ? -xx : ((xx >= width) ? 2*width - xx - 2 : xx);

                    float weight = kernel[(ky+radius)*kernel_size + (kx+radius)];
                    int idx = (yy * width + xx) * 3;
                    sum_r += img[idx] * weight;
                    sum_g += img[idx+1] * weight;
                    sum_b += img[idx+2] * weight;
                }
            }

            // result storing
            int out_idx = (y * width + x) * 3;
            out[out_idx] = clamp((int)roundf(sum_r));
            out[out_idx+1] = clamp((int)roundf(sum_g));
            out[out_idx+2] = clamp((int)roundf(sum_b));
        }
    }

    finalize_and_save("smoothing",output_path, out, width, height, img, start);

   
    return 0;
}