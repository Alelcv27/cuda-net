#include "utils.cuh"

// CUDA kernel for matrix multiplication (A @ B)
__global__ void matmul_a_b_kernel(float* A, float* B, float* C, int m, int n, int k);

// CUDA kernel for matrix multiplication (A @ B.T)
__global__ void matmul_a_bt_kernel(float* A, float* B, float* C, int m, int n, int k);

// CUDA kernel for matrix multiplication (A.T @ B)
__global__ void matmul_at_b_kernel(float* A, float* B, float* C, int m, int n, int k);

// CUDA kernel for bias addition
__global__ void bias_add_kernel(float* x, float* bias, int batch_size, int size);

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* x, int size);

// CUDA kernel for ReLU derivative
__global__ void drelu_kernel(float* x, float* d_ReLU_out, int size);

// CUDA kernel for Leaky ReLU
__global__ void leaky_relu_kernel(float* data, int size);

// CUDA kernel for Leaky ReLU derivative
__global__ void dleaky_relu_kernel(float* input, float* output, int size);

// CUDA kernel for Sigmoid activation
__global__ void sigmoid_kernel(float* x, int size);

// CUDA kernel for Sigmoid derivative
__global__ void dsigmoid_kernel(float* input, float* output, int size);

// CUDA kernel for Softmax
__global__ void softmax_kernel(float* x, int batch_size, int size);

// Add this CUDA kernel to zero out gradients
__global__ void zero_grad_kernel(float* grad, int size);

// CUDA kernel for computing output gradients
__global__ void compute_output_gradients_kernel(float* grad_output, float* output, int* labels, int batch_size);
__global__ void compute_output_gradients_kernel(float* gradients, float* output, int* labels, int batch_size, int output_size);

// CUDA kernel for updating gradients
__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, float* prev_layer, int batch_size, int prev_size, int curr_size);

// Element-wise multiplication of d_dX2 and d_grad_hidden
__global__ void multiply_gradients_kernel(float* grad1, float* grad2, int size);

// Gradient Descent Step
__global__ void update_weights_kernel(float* weights, float* grad_weights, int size);

// Gradient Descent Step (Version for Learning Rate Scheduling)
__global__ void update_weights_kernel(float* weights, float* gradients, int size, float learning_rate);

// Gradient Clipping
__global__ void clip_gradients_kernel(float* gradients, int size, float threshold);

// L2 Regularization
__global__ void update_weights_l2_kernel(float* weights, float* gradients, int size, float learning_rate, float l2_lambda);