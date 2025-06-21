#include "../common/utils.h"


int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_image output_image\n", argv[0]);
        return 1;
    }

    char input_path[512], output_path[512];
    build_paths(argv[1], argv[2], input_path, output_path);

    int width, height;
    unsigned char *img = load_image(input_path, &width, &height);
    if (!img) return 1;

    unsigned char *out = malloc(width * height * 3);
    if (!out) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    clock_t start = clock();

    // Sharpen filter convolution
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

       finalize_and_save("sharpening",output_path, out, width, height, img, start);

    return 0;
}
