#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 512
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.001

// CUDA_CHECK macro to print more information
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)