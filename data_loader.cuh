#pragma once

#include "utils.cuh"

class DataLoader {
public:
    static void load_data(const char* filename, float* data, int size, 
                         bool augment = false, int height = 28, 
                         int width = 28, int channels = 1);
    static void load_labels(const char* filename, int* labels, int size);

private:
    static void random_rotation(float* data, int height, int width, int channels);
    static void random_noise(float* data, int size);
    static void horizontal_flip(float* data, int height, int width, int channels);
};
