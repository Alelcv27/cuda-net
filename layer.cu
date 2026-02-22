#include "layer.cuh"
#include <random>
#include <cmath>

/**
 * Constructor for the Layer class
 * Creates a neural network layer with specified dimensions and activation function
 * 
 * @param input_size Number of input neurons
 * @param output_size Number of output neurons
 * @param activation_type Type of activation function to use
 * @param max_batch_size Maximum batch size for memory allocation
 */
Layer::Layer(int input_size, int output_size, ActivationType activation_type, int max_batch_size) {
	this->input_size = input_size;
	this->output_size = output_size;
	this->activation_type = activation_type;

    // Allocate device memory for weights and biases
    CUDA_CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));

    // Allocate device memory accounting for batch size
    CUDA_CHECK(cudaMalloc(&d_input_values, max_batch_size * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_values, max_batch_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_activated_values, max_batch_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta, max_batch_size * output_size * sizeof(float)));

    // Initialize weights with He initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_size));

    float* h_weights = new float[input_size * output_size];
    float* h_biases = new float[output_size];

    // Initialize weights with He initialization
    for (int i = 0; i < input_size * output_size; ++i) {
        h_weights[i] = dist(gen);
    }

    // Initialize biases to zero
    for (int i = 0; i < output_size; ++i) {
        h_biases[i] = 0.0f;
    }

    // Copy initialized values to device
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, h_biases, output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Free host memory
    delete[] h_weights;
    delete[] h_biases;
}

/**
 * Destructor for the Layer class
 * Frees all allocated CUDA memory
 */
Layer::~Layer() {
    // Free device memory
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_biases));
    CUDA_CHECK(cudaFree(d_input_values));
    CUDA_CHECK(cudaFree(d_output_values));
    CUDA_CHECK(cudaFree(d_activated_values));
    CUDA_CHECK(cudaFree(d_delta));
}

/**
 * Forward pass through the layer
 * Computes output = activation(input * weights + biases)
 * 
 * @param d_input Input data on device memory
 * @param batch_size Number of samples in the current batch
 */

void Layer::forward(float* d_input, int batch_size) {
    // Copy input values
    CUDA_CHECK(cudaMemcpy(d_input_values, d_input, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToDevice));

    // Perform matrix multiplication: output = input * weights
    dim3 block_size(16, 16);
    dim3 grid_size((output_size + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y);

    matmul_a_b_kernel << <grid_size, block_size >> > (d_input, d_weights, d_output_values,
        batch_size, input_size, output_size);

    // Add biases
    bias_add_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_output_values, d_biases, batch_size, output_size);

    // Apply activation function
    if (activation_type == RELU) {
        relu_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
            d_output_values, batch_size * output_size);
    }
    else if (activation_type == SIGMOID) {
        sigmoid_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
            d_output_values, batch_size * output_size);
    }
    else if (activation_type == SOFTMAX) {
        softmax_kernel << <batch_size, 1 >> > (d_output_values, batch_size, output_size);
    }
    else if (activation_type == LEAKY_RELU) {
        leaky_relu_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
            d_output_values, batch_size * output_size);
    }

    // Copy output to activated values
    CUDA_CHECK(cudaMemcpy(d_activated_values, d_output_values,
        batch_size * output_size * sizeof(float), cudaMemcpyDeviceToDevice));

    // Synchronize to ensure computation is complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Sets the delta (error gradient) for this layer directly
 * Typically used for the output layer where error is computed from loss function
 * 
 * @param d_new_delta New delta values on device memory
 * @param batch_size Number of samples in the current batch
 */
void Layer::set_delta(float* d_new_delta, int batch_size) {
    CUDA_CHECK(cudaMemcpy(d_delta, d_new_delta, batch_size * output_size * sizeof(float),
        cudaMemcpyDeviceToDevice));
}

/**
 * Backward pass through the layer
 * Computes the gradient of the loss with respect to this layer's outputs
 * Uses the chain rule: delta = (next_layer_weights * next_layer_delta) * derivative_of_activation
 * 
 * @param d_prev_layer_activated_values Activated values from previous layer
 * @param d_next_layer_delta Delta values from the next layer
 * @param d_next_layer_weights Weights from the next layer
 * @param next_layer_size Size of the next layer
 * @param batch_size Number of samples in the current batch
 */

void Layer::backward(float* d_prev_layer_activated_values, float* d_next_layer_delta,
    float* d_next_layer_weights, int next_layer_size, int batch_size) {

    // Compute this layer's delta based on next layer's delta and weights
    dim3 block_size(16, 16);
    dim3 grid_size((output_size + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y);

    // Calculate delta = (next_layer_weights * next_layer_delta) * derivative_of_activation
    matmul_a_bt_kernel << <grid_size, block_size >> > (d_next_layer_delta, d_next_layer_weights,
        d_delta, batch_size, next_layer_size, output_size);

    // Apply derivative of activation function
    float* d_activation_derivative;
    CUDA_CHECK(cudaMalloc(&d_activation_derivative, batch_size * output_size * sizeof(float)));

    if (activation_type == RELU) {
        drelu_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
            d_output_values, d_activation_derivative, batch_size * output_size);
    }
    else if (activation_type == LEAKY_RELU) {
        dleaky_relu_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
            d_output_values, d_activation_derivative, batch_size * output_size);
    }
    else if (activation_type == SIGMOID) {
        dsigmoid_kernel<<<(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >>>(d_output_values,
            d_activation_derivative, batch_size * output_size);
    }
    // No derivative needed for output layer with softmax + cross-entropy loss

    // Multiply delta by derivative of activation function
    multiply_gradients_kernel << <(batch_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_delta, d_activation_derivative, batch_size * output_size);

    CUDA_CHECK(cudaFree(d_activation_derivative));
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Updates the weights and biases based on computed gradients
 * Implements gradient descent: weights -= learning_rate * gradients
 * 
 * @param d_input Input data used for computing weight gradients
 * @param batch_size Number of samples in the current batch
 * @param learning_rate Learning rate for gradient descent
 */

void Layer::update_weights(float* d_input, int batch_size, float learning_rate) {

    dim3 block_size(16, 16);
    dim3 grid_size((input_size + block_size.x - 1) / block_size.x,
        (output_size + block_size.y - 1) / block_size.y);

    // Compute weight gradients: grad_w = input.T * delta
    float* d_grad_weights;
    float* d_grad_biases;

    CUDA_CHECK(cudaMalloc(&d_grad_weights, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_biases, output_size * sizeof(float)));

    // Initialize gradients to zero
    zero_grad_kernel << <(input_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_grad_weights, input_size * output_size);
    zero_grad_kernel << <(output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_grad_biases, output_size);

    // Compute gradients
    matmul_at_b_kernel << <grid_size, block_size >> > (d_input, d_delta, d_grad_weights,
        batch_size, input_size, output_size);

    // Update gradients for bias
    update_gradients_kernel << <grid_size, block_size >> > (d_grad_weights, d_grad_biases,
        d_delta, d_input, batch_size, input_size, output_size);

    // Update weights and biases with learning rate
    update_weights_kernel << <(input_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_weights, d_grad_weights, input_size * output_size, learning_rate);
    update_weights_kernel << <(output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_biases, d_grad_biases, output_size, learning_rate);

    // Free temporary memory
    CUDA_CHECK(cudaFree(d_grad_weights));
    CUDA_CHECK(cudaFree(d_grad_biases));

    CUDA_CHECK(cudaDeviceSynchronize());
}


void Layer::update_weights_l2(float* d_input, int batch_size, float learning_rate, float l2_lambda) {

    dim3 block_size(16, 16);
    dim3 grid_size((input_size + block_size.x - 1) / block_size.x,
        (output_size + block_size.y - 1) / block_size.y);

    // Compute weight gradients: grad_w = input.T * delta
    float* d_grad_weights;
    float* d_grad_biases;

    CUDA_CHECK(cudaMalloc(&d_grad_weights, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_biases, output_size * sizeof(float)));

    // Initialize gradients to zero
    zero_grad_kernel << <(input_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_grad_weights, input_size * output_size);
    zero_grad_kernel << <(output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_grad_biases, output_size);

    // Compute gradients
    matmul_at_b_kernel << <grid_size, block_size >> > (d_input, d_delta, d_grad_weights,
        batch_size, input_size, output_size);

    // Update gradients for bias
    update_gradients_kernel << <grid_size, block_size >> > (d_grad_weights, d_grad_biases,
        d_delta, d_input, batch_size, input_size, output_size);

    // Update weights and biases with learning rate
    update_weights_l2_kernel << <(input_size * output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_weights, d_grad_weights, input_size * output_size, learning_rate, l2_lambda);
    update_weights_l2_kernel << <(output_size + block_size.x - 1) / block_size.x, block_size.y >> > (
        d_biases, d_grad_biases, output_size, learning_rate, l2_lambda);

    // Free temporary memory
    CUDA_CHECK(cudaFree(d_grad_weights));
    CUDA_CHECK(cudaFree(d_grad_biases));

    CUDA_CHECK(cudaDeviceSynchronize());
}