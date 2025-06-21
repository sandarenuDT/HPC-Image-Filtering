// image_utils.h
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include "../stb_image.h"
#include "../stb_image_write.h"
#include <time.h>
// #include <omp.h>

// Clamp pixel values to [0, 255]
int clamp(int val);

// Load an image from file
unsigned char* load_image(const char* input_path, int* width, int* height);

// Save an image to file
int save_image(const char* output_path, unsigned char* data, int width, int height);

// Finalize and save for serial version
void finalize_and_save(const char *filter_name, const char *output_path, unsigned char *out,
                       int width, int height, unsigned char *original, clock_t start);
                       
// Build file paths for input and output
void build_paths(const char* input_filename, const char* output_filename, char* input_path, char* output_path);

#endif
