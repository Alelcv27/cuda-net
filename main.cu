#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "utils.cuh"
#include "kernels.cuh"
#include "data_loader.cuh"
#include "neural_network.cuh"


int main() {
    std::cout << "CUDA Neural Network" << std::endl;
    std::cout << "===================" << std::endl;

    // Load MNIST data
    float* X_train = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int* y_train = (int*)malloc(TRAIN_SIZE * sizeof(int));
    float* X_test = (float*)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int* y_test = (int*)malloc(TEST_SIZE * sizeof(int));

    DataLoader dataLoader;
    std::cout << "Loading training data..." << std::endl;
    dataLoader.load_data("X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE, 
                        true,  // Enable augmentation for training data
                        28,
                        28,
                        1);
    dataLoader.load_labels("y_train.bin", y_train, TRAIN_SIZE);

    std::cout << "Loading test data..." << std::endl;
    dataLoader.load_data("X_test.bin", X_test, TEST_SIZE * INPUT_SIZE,
                        false,  // No augmentation for test data
                        28, 28, 1);
    dataLoader.load_labels("y_test.bin", y_test, TEST_SIZE);

    std::cout << "Data loaded successfully." << std::endl;

    // Create neural network
    NeuralNetwork nn(BATCH_SIZE, LEARNING_RATE);

    // Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
    nn.add_layer(INPUT_SIZE, HIDDEN1_SIZE, LEAKY_RELU);
    nn.add_layer(HIDDEN1_SIZE, HIDDEN2_SIZE, LEAKY_RELU);
    nn.add_layer(HIDDEN2_SIZE, OUTPUT_SIZE, SOFTMAX);

    // Train the network
    std::cout << "Starting training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    nn.train_early_with_lr(X_train, y_train, TRAIN_SIZE, X_test, y_test, TEST_SIZE, EPOCHS);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Training completed." << std::endl;
    std::cout << "Total training time: " << duration.count() / 1000.0 << " seconds" << std::endl;

    // Evaluate final accuracy
    float final_accuracy = nn.evaluate_accuracy(X_test, y_test, TEST_SIZE);
    std::cout << "Final test accuracy: " << final_accuracy << "%" << std::endl;

    // Free host memory
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
