#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include "layer.cuh"
#include <vector>

class NeuralNetwork {
private:
    std::vector<Layer*> layers;
    float learning_rate;
    int batch_size;
    float l2_lambda;

    float* d_input;
    int* d_labels;
    float* d_output_gradients;

public:
    NeuralNetwork(int batch_size, float learning_rate, float l2_lambda = 0.01f);
    ~NeuralNetwork();

    void add_layer(int input_size, int output_size, ActivationType activation_type);
    void backward(int* labels);
    void update_weights();
    float compute_loss(int* labels);
    void forward(float* input_data, int current_batch_size = 0);
    void train(float* train_data, int* train_labels, int train_size, float* test_data, int* test_labels, int test_size,
        int epochs);
    float evaluate_accuracy(float* test_data, int* test_labels, int test_size);
    void train_early(float* train_data, int* train_labels, int train_size, float* test_data, int* test_labels, int test_size,
                         int epochs, int patience = 5, float min_delta = 0.001f);
    void train_early_with_lr(float* train_data, int* train_labels, int train_size, float* test_data, int* test_labels, int test_size,
                                 int epochs, int patience = 5, float min_delta = 0.001f, float lr_reduction_factor = 0.5f, int lr_patience = 2);

    // Getters
    float* get_output() const;
    int get_input_size() const;
    int get_output_size() const;
    int get_batch_size() const { return batch_size; }
};

#endif // NEURAL_NETWORK_CUH
