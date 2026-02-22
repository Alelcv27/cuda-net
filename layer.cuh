#ifndef LAYER_CUH
#define LAYER_CUH

#include <cuda_runtime.h>
#include "kernels.cuh"
#include "utils.cuh"

enum ActivationType {
    RELU,
    SIGMOID,
    SOFTMAX,
    LEAKY_RELU
};

class Layer {
private:
    int input_size;
    int output_size;
    float* d_weights;
    float* d_biases;
    float* d_input_values;
    float* d_output_values;
    float* d_activated_values;
    float* d_delta;
    ActivationType activation_type;

public:
    Layer(int input_size, int output_size, ActivationType activation_type, int max_batch_size = 128);
    ~Layer();

    void forward(float* d_input, int batch_size);
    void backward(float* d_prev_layer_activated_values, float* d_next_layer_delta,
        float* d_next_layer_weights, int next_layer_size, int batch_size);
    void update_weights(float* d_input, int batch_size, float learning_rate);
    void update_weights_l2(float* input, int batch_size, float learning_rate, float l2_lambda);

    // Getter methods
    float* get_weights() const { return d_weights; }
    float* get_biases() const { return d_biases; }
    float* get_activated_values() const { return d_activated_values; }
    float* get_output_values() const { return d_output_values; }
    float* get_delta() const { return d_delta; }
    int get_input_size() const { return input_size; }
    int get_output_size() const { return output_size; }

    // Setter methods
    void set_delta(float* d_new_delta, int batch_size);
};

#endif // LAYER_CUH