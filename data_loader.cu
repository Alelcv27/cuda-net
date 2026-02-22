#include "data_loader.cuh"
#include <random>
#include <cmath>
#include <corecrt_math_defines.h>

// Random number generator for augmentations
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> dis(-1.0, 1.0);

void DataLoader::random_rotation(float* data, int height, int width, int channels) {
    // Maximum rotation angle in degrees
    const float max_angle = 15.0f;
    float angle = dis(gen) * max_angle;
    float radian = angle * M_PI / 180.0f;
    float cos_theta = cos(radian);
    float sin_theta = sin(radian);

    float* temp = new float[height * width * channels];
    memcpy(temp, data, height * width * channels * sizeof(float));

    int center_x = width / 2;
    int center_y = height / 2;

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Compute rotated coordinates
                int rx = center_x + (x - center_x) * cos_theta - (y - center_y) * sin_theta;
                int ry = center_y + (x - center_x) * sin_theta + (y - center_y) * cos_theta;

                // Check bounds and copy pixel
                if (rx >= 0 && rx < width && ry >= 0 && ry < height) {
                    data[(y * width + x) * channels + c] = 
                        temp[(ry * width + rx) * channels + c];
                } else {
                    data[(y * width + x) * channels + c] = 0.0f;
                }
            }
        }
    }
    delete[] temp;
}

void DataLoader::random_noise(float* data, int size) {
    std::normal_distribution<float> noise(0.0f, 0.01f);
    for (int i = 0; i < size; i++) {
        data[i] += noise(gen);
        data[i] = std::min(std::max(data[i], 0.0f), 1.0f);  // Clip values
    }
}

void DataLoader::horizontal_flip(float* data, int height, int width, int channels) {
    std::uniform_real_distribution<float> flip_prob(0.0f, 1.0f);
    if (flip_prob(gen) > 0.5f) {  // 50% chance to flip
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width / 2; x++) {
                for (int c = 0; c < channels; c++) {
                    int idx1 = (y * width + x) * channels + c;
                    int idx2 = (y * width + (width - 1 - x)) * channels + c;
                    std::swap(data[idx1], data[idx2]);
                }
            }
        }
    }
}

void DataLoader::load_data(const char* filename, float* data, int size, bool augment, 
                         int height, int width, int channels) {
    // Original data loading
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);

    // Apply augmentations if requested
    if (augment) {
        // For CIFAR-10 (RGB), apply horizontal flip
        if (channels == 3) {
            horizontal_flip(data, height, width, channels);
        }
        
        // Apply rotation (works for both MNIST and CIFAR-10)
        random_rotation(data, height, width, channels);
        
        // Apply random noise
        random_noise(data, size);
    }
}

// load batch labels
void DataLoader::load_labels(const char* filename, int* labels, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}