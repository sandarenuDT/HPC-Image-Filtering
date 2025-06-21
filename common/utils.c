#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils.h"

int clamp(int val) {
    return (val < 0) ? 0 : ((val > 255) ? 255 : val);
}

unsigned char* load_image(const char* input_path, int* width, int* height) {
    int channels;
    unsigned char* img = stbi_load(input_path, width, height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error loading image %s\n", input_path);
    }
    return img;
}


int save_image(const char* output_path, unsigned char* data, int width, int height) {
    if (!stbi_write_png(output_path, width, height, 3, data, width * 3)) {
        fprintf(stderr, "Error saving image %s\n", output_path);
        return 0;
    }
    printf("Image saved to %s\n", output_path);
    return 1;
}
void finalize_and_save(const char *filter_name, const char *output_path, unsigned char *out, int width, int height,
                       unsigned char *original, clock_t start) {
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s took %.4f seconds\n", filter_name, elapsed);

    if (!stbi_write_png(output_path, width, height, 3, out, width * 3)) {
        fprintf(stderr, "Error saving image %s\n", output_path);
    } else {
        printf("Image saved to %s\n", output_path);
    }

    free(out);
    stbi_image_free(original);
}


void build_paths(const char* input_filename, const char* output_filename, char* input_path, char* output_path) {
    snprintf(input_path, 512, "../inputImages/%s", input_filename);
    snprintf(output_path, 512, "../outputImages/%s", output_filename);
}

