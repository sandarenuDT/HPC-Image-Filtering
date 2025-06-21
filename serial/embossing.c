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

    clock_t start = clock();

    // Emboss filter
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

        finalize_and_save("embossing",output_path, out, width, height, img, start);
    return 0;
}
