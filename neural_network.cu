#include "neural_network.cuh"
#include <iostream>
#include <cmath>

/**
 * Constructor for the NeuralNetwork class.
 * Initializes a neural network with the specified batch size and learning rate.
 * 
 * @param batch_size Number of samples to process in each training iteration
 * @param learning_rate Step size for gradient descent optimization
 */
NeuralNetwork::NeuralNetwork(int batch_size, float learning_rate, float l2_lambda)
    : batch_size(batch_size), learning_rate(learning_rate), l2_lambda(l2_lambda) {
}

/**
 * Destructor for the NeuralNetwork class.
 * Frees all allocated GPU memory and deletes layer objects.
 */
NeuralNetwork::~NeuralNetwork() {
    if (d_input) CUDA_CHECK(cudaFree(d_input));
    if (d_labels) CUDA_CHECK(cudaFree(d_labels));
    if (d_output_gradients) CUDA_CHECK(cudaFree(d_output_gradients));

    // Free layer memory
    for (Layer* layer : layers) {
        delete layer;
    }
}

/**
 * Adds a new layer to the neural network.
 * Creates a new layer with the specified dimensions and activation function.
 * Also allocates necessary GPU memory for the first layer's input and output gradients.
 * 
 * @param input_size Number of input neurons for this layer
 * @param output_size Number of output neurons for this layer
 * @param activation_type Type of activation function to use (e.g., RELU, SIGMOID, SOFTMAX)
 */

void NeuralNetwork::add_layer(int input_size, int output_size, ActivationType activation_type) {
    Layer* layer = new Layer(input_size, output_size, activation_type, batch_size);
    layers.push_back(layer);

    // Allocate input buffer for the first layer
    if (layers.size() == 1) {
        CUDA_CHECK(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, batch_size * sizeof(int)));
    }

    CUDA_CHECK(cudaMalloc(&d_output_gradients, batch_size * output_size * sizeof(float)));
}

/**
 * Performs a forward pass with support for variable batch sizes.
 * Handles cases where the current batch size is smaller than the configured batch size.
 * Zeros out unused portions of the input buffer to prevent garbage data from affecting computation.
 *
 * @param input_data Pointer to input data in host memory
 * @param current_batch_size Number of samples in the current batch
 */

void NeuralNetwork::forward(float* input_data, int current_batch_size) {
    // If no custom batch size specified, use the default
    if (current_batch_size <= 0 || current_batch_size > batch_size) {
        current_batch_size = batch_size;
    }

    // Only copy the actual data we have (current_batch_size elements)
    CUDA_CHECK(cudaMemcpy(d_input, input_data,
        current_batch_size * layers[0]->get_input_size() * sizeof(float),
        cudaMemcpyHostToDevice));

    // If the current batch is smaller than the full batch size, zero out the rest
    // This prevents garbage data from affecting the computation
    if (current_batch_size < batch_size) {
        size_t offset = current_batch_size * layers[0]->get_input_size() * sizeof(float);
        size_t remaining = (batch_size - current_batch_size) * layers[0]->get_input_size() * sizeof(float);
        CUDA_CHECK(cudaMemset((char*)d_input + offset, 0, remaining));
    }

    // Forward pass
    float* layer_input = d_input;
    for (Layer* layer : layers) {
        layer->forward(layer_input, batch_size);
        layer_input = layer->get_activated_values();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}


/**
 * Computes the cross-entropy loss between network outputs and true labels.
 * Copies labels to GPU, retrieves network outputs, and calculates the average loss.
 * 
 * @param labels Pointer to integer labels in host memory
 * @return Average cross-entropy loss across the batch
 */

float NeuralNetwork::compute_loss(int* labels) {
    // Copy labels to device
    CUDA_CHECK(cudaMemcpy(d_labels, labels, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // Get output layer's activated values
    float* d_output = layers.back()->get_activated_values();
    int output_size = layers.back()->get_output_size();

    // Copy output to host for loss calculation
    float* h_output = new float[batch_size * output_size];
    int* h_labels = new int[batch_size];

    if (h_output == nullptr || h_labels == nullptr) {
        throw std::runtime_error("Failed to allocate host memory for loss calculation");
    }

    CUDA_CHECK(cudaMemcpy(h_output, d_output,
        batch_size * output_size * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_labels, d_labels, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate cross-entropy loss
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int label = h_labels[b];
        // Make sure label is within bounds
        if (label < 0 || label >= output_size) {
            std::cerr << "Warning: Label " << label << " out of bounds (output size: " << output_size << ")" << std::endl;
            continue; // Skip this sample
        }
        float prob = h_output[b * output_size + label];
        total_loss -= log(fmax(prob, 1e-7f)); // Prevent log(0)
    }

    delete[] h_output;
    delete[] h_labels;

    return total_loss / batch_size;
}

/**
 * Performs the backward pass (backpropagation) through the network.
 * Computes gradients starting from the output layer and propagates them backward.
 * 
 * @param labels Pointer to integer labels in host memory
 */

void NeuralNetwork::backward(int* labels) {
    // Copy labels to device
    CUDA_CHECK(cudaMemcpy(d_labels, labels, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate output layer gradients
    Layer* output_layer = layers.back();
    float* d_output = output_layer->get_activated_values();
    int output_size = output_layer->get_output_size();

    // Compute gradients for output layer (assuming softmax + cross-entropy)
    compute_output_gradients_kernel << <(batch_size + 255) / 256, 256 >> > (
        d_output_gradients, d_output, d_labels, batch_size, output_size);

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Set output layer delta
    output_layer->set_delta(d_output_gradients, batch_size);

    // Backpropagate through hidden layers
    for (int i = layers.size() - 2; i >= 0; i--) {
        Layer* current_layer = layers[i];
        Layer* next_layer = layers[i + 1];

        float* prev_layer_output = (i > 0) ? layers[i - 1]->get_activated_values() : d_input;

        current_layer->backward(
            prev_layer_output,
            next_layer->get_delta(),
            next_layer->get_weights(),
            next_layer->get_output_size(),
            batch_size
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}



/**
 * Updates the weights of all layers using computed gradients.
 * Applies gradient descent with gradient clipping and L2 regularization.
 */

void NeuralNetwork::update_weights() {
    
    // Update weights for all layers
    for (size_t i = 0; i < layers.size(); i++) {
        float* layer_input = (i == 0) ? d_input : layers[i - 1]->get_activated_values();
        Layer* layer = layers[i];

        //layer->update_weights_l2(layer_input, batch_size, learning_rate, l2_lambda);
        layer->update_weights(layer_input, batch_size, learning_rate);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Returns the input size of the network (input size of the first layer).
 * 
 * @return Input size or 0 if no layers exist
 */
int NeuralNetwork::get_input_size() const {
    if (layers.empty()) {
        return 0;
    }
    return layers[0]->get_input_size();
}

/**
 * Returns the output size of the network (output size of the last layer).
 * 
 * @return Output size or 0 if no layers exist
 */
int NeuralNetwork::get_output_size() const {
    if (layers.empty()) {
        return 0;
    }
    return layers.back()->get_output_size();
}

/**
 * Returns a pointer to the output values of the network.
 * 
 * @return Pointer to output values in device memory, or nullptr if no layers exist
 */
float* NeuralNetwork::get_output() const {
    if (layers.empty()) {
        return nullptr;
    }
    return layers.back()->get_activated_values();
}



/**
 * Training method that safely handles variable batch sizes.
 * Similar to train() but properly handles the last batch which may be smaller than batch_size.
 * 
 * @param train_data Pointer to training input data in host memory
 * @param train_labels Pointer to training labels in host memory
 * @param train_size Number of training samples
 * @param test_data Pointer to test input data in host memory
 * @param test_labels Pointer to test labels in host memory
 * @param test_size Number of test samples
 * @param epochs Number of complete passes through the training data
 */
void NeuralNetwork::train(float* train_data, int* train_labels, int train_size,
    float* test_data, int* test_labels, int test_size,
    int epochs) {
    // Calculate number of full and partial batches
    int num_batches = (train_size + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            // Calculate the actual batch size for this iteration
            int current_batch_size = std::min(batch_size, train_size - b * batch_size);
            if (current_batch_size <= 0) break;

            // Get pointers to the current batch data
            float* batch_input = &train_data[b * batch_size * get_input_size()];
            int* batch_labels = &train_labels[b * batch_size];

            // Forward pass with safe method that handles partial batches
            forward(batch_input, current_batch_size);

            // Copy labels (only the actual number we have)
            CUDA_CHECK(cudaMemcpy(d_labels, batch_labels,
                current_batch_size * sizeof(int),
                cudaMemcpyHostToDevice));

            // If we have a partial batch, zero out the rest of the labels
            if (current_batch_size < batch_size) {
                CUDA_CHECK(cudaMemset((char*)d_labels + (current_batch_size * sizeof(int)),
                    0, (batch_size - current_batch_size) * sizeof(int)));
            }

            // Compute loss (adjusted for partial batch)
            float* d_output = layers.back()->get_activated_values();
            int output_size = layers.back()->get_output_size();

            float* h_output = new float[current_batch_size * output_size];
            int* h_labels = new int[current_batch_size];

            if (h_output == nullptr || h_labels == nullptr) {
                throw std::runtime_error("Failed to allocate host memory for loss calculation");
            }

            CUDA_CHECK(cudaMemcpy(h_output, d_output,
                current_batch_size * output_size * sizeof(float),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_labels, d_labels,
                current_batch_size * sizeof(int),
                cudaMemcpyDeviceToHost));

            // Calculate cross-entropy loss
            float batch_loss = 0.0f;
            for (int i = 0; i < current_batch_size; i++) {
                int label = h_labels[i];
                if (label < 0 || label >= output_size) {
                    std::cerr << "Warning: Label " << label << " out of bounds (output size: "
                        << output_size << ")" << std::endl;
                    continue;
                }
                float prob = h_output[i * output_size + label];
                batch_loss -= log(fmax(prob, 1e-7f));
            }
            batch_loss /= current_batch_size;
            total_loss += batch_loss;

            delete[] h_output;
            delete[] h_labels;

            // Backward pass
            backward(batch_labels);

            // Update weights
            update_weights();

            // Print progress
            if ((b + 1) % 100 == 0) {
                // Use a smaller batch for evaluation if needed
                int eval_batch = std::min(batch_size, std::min(test_size, 1000));
                float accuracy = evaluate_accuracy(test_data, test_labels, eval_batch);

                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                    << ", Batch " << (b + 1) << "/" << num_batches
                    << ", Loss: " << (total_loss / (b + 1))
                    << ", Accuracy: " << accuracy << "%" << std::endl;
            }
        }

        // Evaluate on full test set at end of epoch
        float accuracy = evaluate_accuracy(test_data, test_labels, test_size);
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
            << " completed. Average loss: " << (total_loss / num_batches)
            << ", Test accuracy: " << accuracy << "%" << std::endl;
    }
}

/**
 * Enhanced evaluation method that safely handles variable batch sizes.
 * 
 * @param test_data Pointer to test input data in host memory
 * @param test_labels Pointer to test labels in host memory
 * @param test_size Number of test samples
 * @return Accuracy as a percentage (0-100)
 */
float NeuralNetwork::evaluate_accuracy(float* test_data, int* test_labels, int test_size) {
    int num_batches = (test_size + batch_size - 1) / batch_size;
    int correct = 0;
    int total = 0;

    for (int b = 0; b < num_batches; b++) {
        int current_batch_size = std::min(batch_size, test_size - b * batch_size);
        if (current_batch_size <= 0) break;

        // Forward pass with safe method
        forward(&test_data[b * batch_size * get_input_size()], current_batch_size);

        // Get predictions
        float* d_output = layers.back()->get_activated_values();
        int output_size = layers.back()->get_output_size();

        float* h_output = new float[current_batch_size * output_size];
        int* h_labels = new int[current_batch_size];

        if (h_output == nullptr || h_labels == nullptr) {
            throw std::runtime_error("Failed to allocate host memory for accuracy evaluation");
        }

        CUDA_CHECK(cudaMemcpy(h_output, d_output,
            current_batch_size * output_size * sizeof(float),
            cudaMemcpyDeviceToHost));

        // Copy labels from host to host (just a regular memcpy)
        memcpy(h_labels, &test_labels[b * batch_size], current_batch_size * sizeof(int));

        // Count correct predictions
        for (int i = 0; i < current_batch_size; i++) {
            int pred_label = 0;
            for (int j = 1; j < output_size; j++) {
                if (h_output[i * output_size + j] > h_output[i * output_size + pred_label]) {
                    pred_label = j;
                }
            }

            if (pred_label == h_labels[i]) {
                correct++;
            }
            total++;
        }

        delete[] h_output;
        delete[] h_labels;
    }

    return 100.0f * correct / total;
}

/**
 * Training method with early stopping capability.
 * Monitors validation metrics and stops training if no improvement is seen for a specified number of epochs.
 * 
 * @param train_data Pointer to training input data in host memory
 * @param train_labels Pointer to training labels in host memory
 * @param train_size Number of training samples
 * @param test_data Pointer to test input data in host memory
 * @param test_labels Pointer to test labels in host memory
 * @param test_size Number of test samples
 * @param epochs Maximum number of complete passes through the training data
 * @param patience Number of epochs with no improvement after which training will be stopped
 * @param min_delta Minimum change in monitored metrics to qualify as an improvement
 */
void NeuralNetwork::train_early(float* train_data, int* train_labels, int train_size,
    float* test_data, int* test_labels, int test_size,
    int epochs, int patience, float min_delta) {
    // Calculate number of full and partial batches
    int num_batches = (train_size + batch_size - 1) / batch_size;
    
    // Early stopping variables
    float best_accuracy = 0.0f;
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    
    // Degradation thresholds
    const float max_accuracy_drop = 10.0f;  // Stop if accuracy drops by 10% from best
    const float max_loss_increase = 0.5f;   // Stop if loss increases by 0.5 from best
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            // Calculate the actual batch size for this iteration
            int current_batch_size = std::min(batch_size, train_size - b * batch_size);
            if (current_batch_size <= 0) break;

            // Get pointers to the current batch data
            float* batch_input = &train_data[b * batch_size * get_input_size()];
            int* batch_labels = &train_labels[b * batch_size];

            // Forward pass with safe method that handles partial batches
            forward(batch_input, current_batch_size);

            // Copy labels (only the actual number we have)
            CUDA_CHECK(cudaMemcpy(d_labels, batch_labels,
                current_batch_size * sizeof(int),
                cudaMemcpyHostToDevice));

            // If we have a partial batch, zero out the rest of the labels
            if (current_batch_size < batch_size) {
                CUDA_CHECK(cudaMemset((char*)d_labels + (current_batch_size * sizeof(int)),
                    0, (batch_size - current_batch_size) * sizeof(int)));
            }

            // Compute loss (adjusted for partial batch)
            float* d_output = layers.back()->get_activated_values();
            int output_size = layers.back()->get_output_size();

            float* h_output = new float[current_batch_size * output_size];
            int* h_labels = new int[current_batch_size];

            if (h_output == nullptr || h_labels == nullptr) {
                throw std::runtime_error("Failed to allocate host memory for loss calculation");
            }

            CUDA_CHECK(cudaMemcpy(h_output, d_output,
                current_batch_size * output_size * sizeof(float),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_labels, d_labels,
                current_batch_size * sizeof(int),
                cudaMemcpyDeviceToHost));

            // Calculate cross-entropy loss
            float batch_loss = 0.0f;
            for (int i = 0; i < current_batch_size; i++) {
                int label = h_labels[i];
                if (label < 0 || label >= output_size) {
                    std::cerr << "Warning: Label " << label << " out of bounds (output size: "
                        << output_size << ")" << std::endl;
                    continue;
                }
                float prob = h_output[i * output_size + label];
                batch_loss -= log(fmax(prob, 1e-7f));
            }
            batch_loss /= current_batch_size;
            total_loss += batch_loss;

            delete[] h_output;
            delete[] h_labels;

            // Backward pass
            backward(batch_labels);

            // Update weights
            update_weights();

            // Print progress
            if ((b + 1) % 100 == 0) {
                // Use a smaller batch for evaluation if needed
                int eval_batch = std::min(batch_size, std::min(test_size, 1000));
                float accuracy = evaluate_accuracy(test_data, test_labels, eval_batch);

                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                    << ", Batch " << (b + 1) << "/" << num_batches
                    << ", Loss: " << (total_loss / (b + 1))
                    << ", Accuracy: " << accuracy << "%" << std::endl;
            }
        }

        // Evaluate on full test set at end of epoch
        float accuracy = evaluate_accuracy(test_data, test_labels, test_size);
        float avg_loss = total_loss / num_batches;
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
            << " completed. Average loss: " << avg_loss
            << ", Test accuracy: " << accuracy << "%" << std::endl;
            
        // Early stopping check
        bool improved = false;
        
        // Check if accuracy improved
        if (accuracy > best_accuracy + min_delta) {
            best_accuracy = accuracy;
            improved = true;
            std::cout << "Accuracy improved to " << best_accuracy << "%" << std::endl;
        }
        
        // Check if loss improved
        if (avg_loss < best_loss - min_delta) {
            best_loss = avg_loss;
            improved = true;
            std::cout << "Loss improved to " << best_loss << std::endl;
        }
        
        // Check for significant performance degradation
        if (accuracy < best_accuracy - max_accuracy_drop) {
            std::cout << "Early stopping: Accuracy dropped by more than " 
                     << max_accuracy_drop << "% from best (" 
                     << best_accuracy << "% -> " << accuracy << "%)" << std::endl;
            break;
        }
        
        if (avg_loss > best_loss + max_loss_increase) {
            std::cout << "Early stopping: Loss increased by more than "
                     << max_loss_increase << " from best ("
                     << best_loss << " -> " << avg_loss << ")" << std::endl;
            break;
        }
        
        // Update patience counter
        if (!improved) {
            patience_counter++;
            std::cout << "No improvement detected. Patience: " << patience_counter << "/" << patience << std::endl;
            
            if (patience_counter >= patience) {
                std::cout << "Early stopping triggered after " << (epoch + 1) << " epochs" << std::endl;
                break;
            }
        } else {
            patience_counter = 0;
        }
    }
    
    // Print final best metrics
    std::cout << "\n=== Training Completed ===\n"
              << "Best accuracy: " << best_accuracy << "%\n"
              << "Best loss: " << best_loss << "\n"
              << "======================" << std::endl;
}

/**
 * Advanced training method with early stopping and learning rate scheduling.
 * Monitors validation metrics and:
 * 1. Reduces learning rate when no improvement is seen for lr_patience epochs
 * 2. Stops training when no improvement is seen for patience epochs
 * 
 * @param train_data Pointer to training input data in host memory
 * @param train_labels Pointer to training labels in host memory
 * @param train_size Number of training samples
 * @param test_data Pointer to test input data in host memory
 * @param test_labels Pointer to test labels in host memory
 * @param test_size Number of test samples
 * @param epochs Maximum number of complete passes through the training data
 * @param patience Number of epochs with no improvement after which training will be stopped
 * @param min_delta Minimum change in monitored metrics to qualify as an improvement
 * @param lr_reduction_factor Factor by which to reduce learning rate (e.g., 0.1 for 10x reduction)
 * @param lr_patience Number of epochs with no improvement after which learning rate will be reduced
 */
void NeuralNetwork::train_early_with_lr(float* train_data, int* train_labels, int train_size,
    float* test_data, int* test_labels, int test_size,
    int epochs, int patience, float min_delta, 
    float lr_reduction_factor, int lr_patience) {
    // Calculate number of full and partial batches
    int num_batches = (train_size + batch_size - 1) / batch_size;
    
    // Early stopping variables
    float best_accuracy = 0.0f;
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    int lr_patience_counter = 0;
    float current_lr = learning_rate;
    
    // Degradation thresholds
    const float max_accuracy_drop = 10.0f;  // Stop if accuracy drops by 10% from best
    const float max_loss_increase = 0.5f;   // Stop if loss increases by 0.5 from best
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            // Calculate the actual batch size for this iteration
            int current_batch_size = std::min(batch_size, train_size - b * batch_size);
            if (current_batch_size <= 0) break;

            // Get pointers to the current batch data
            float* batch_input = &train_data[b * batch_size * get_input_size()];
            int* batch_labels = &train_labels[b * batch_size];

            // Forward pass with safe method that handles partial batches
            forward(batch_input, current_batch_size);

            // Copy labels (only the actual number we have)
            CUDA_CHECK(cudaMemcpy(d_labels, batch_labels,
                current_batch_size * sizeof(int),
                cudaMemcpyHostToDevice));

            // If we have a partial batch, zero out the rest of the labels
            if (current_batch_size < batch_size) {
                CUDA_CHECK(cudaMemset((char*)d_labels + (current_batch_size * sizeof(int)),
                    0, (batch_size - current_batch_size) * sizeof(int)));
            }

            // Compute loss (adjusted for partial batch)
            float* d_output = layers.back()->get_activated_values();
            int output_size = layers.back()->get_output_size();

            float* h_output = new float[current_batch_size * output_size];
            int* h_labels = new int[current_batch_size];

            if (h_output == nullptr || h_labels == nullptr) {
                throw std::runtime_error("Failed to allocate host memory for loss calculation");
            }

            CUDA_CHECK(cudaMemcpy(h_output, d_output,
                current_batch_size * output_size * sizeof(float),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_labels, d_labels,
                current_batch_size * sizeof(int),
                cudaMemcpyDeviceToHost));

            // Calculate cross-entropy loss
            float batch_loss = 0.0f;
            for (int i = 0; i < current_batch_size; i++) {
                int label = h_labels[i];
                if (label < 0 || label >= output_size) {
                    std::cerr << "Warning: Label " << label << " out of bounds (output size: "
                        << output_size << ")" << std::endl;
                    continue;
                }
                float prob = h_output[i * output_size + label];
                batch_loss -= log(fmax(prob, 1e-7f));
            }
            batch_loss /= current_batch_size;
            total_loss += batch_loss;

            delete[] h_output;
            delete[] h_labels;

            // Backward pass
            backward(batch_labels);

			update_weights();

            // Print progress
            if ((b + 1) % 100 == 0) {
                // Use a smaller batch for evaluation if needed
                int eval_batch = std::min(batch_size, std::min(test_size, 1000));
                float accuracy = evaluate_accuracy(test_data, test_labels, eval_batch);

                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                    << ", Batch " << (b + 1) << "/" << num_batches
                    << ", Loss: " << (total_loss / (b + 1))
                    << ", Accuracy: " << accuracy << "%"
                    << ", LR: " << current_lr << std::endl;
            }
        }

        // Evaluate on full test set at end of epoch
        float accuracy = evaluate_accuracy(test_data, test_labels, test_size);
        float avg_loss = total_loss / num_batches;
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
            << " completed. Average loss: " << avg_loss
            << ", Test accuracy: " << accuracy << "%"
            << ", LR: " << current_lr << std::endl;
            
        // Early stopping and learning rate scheduling check
        bool improved = false;
        
        // Check if loss improved
        if (avg_loss < best_loss - min_delta) {
            best_loss = avg_loss;
            improved = true;
            std::cout << "Loss improved to " << best_loss << std::endl;
        }

        // Check if accuracy improved
        if (accuracy > best_accuracy + min_delta * 100) {
            best_accuracy = accuracy;
            improved = true;
            std::cout << "Accuracy improved to " << best_accuracy << "%" << std::endl;
        }
        
        // Check for significant performance degradation
        if (accuracy < best_accuracy - max_accuracy_drop) {
            std::cout << "Early stopping: Accuracy dropped by more than " 
                     << max_accuracy_drop << "% from best (" 
                     << best_accuracy << "% -> " << accuracy << "%)" << std::endl;
            break;
        }
        
        if (avg_loss > best_loss + max_loss_increase) {
            std::cout << "Early stopping: Loss increased by more than "
                     << max_loss_increase << " from best ("
                     << best_loss << " -> " << avg_loss << ")" << std::endl;
            break;
        }
        
        // Update patience counters
        if (!improved) {
            patience_counter++;
            lr_patience_counter++;
            std::cout << "No improvement detected. Patience: " << patience_counter << "/" << patience 
                      << ", LR patience: " << lr_patience_counter << "/" << lr_patience << std::endl;
            
            // Check if we should reduce learning rate
            if (lr_patience_counter >= lr_patience) {
                current_lr *= lr_reduction_factor;
                lr_patience_counter = 0;
                std::cout << "Reducing learning rate to " << current_lr << std::endl;
                
                // If learning rate becomes too small, stop training
                if (current_lr < 1e-6f) {
                    std::cout << "Learning rate too small. Stopping training." << std::endl;
                    break;
                }
            }
            
            // Check if we should stop training
            if (patience_counter >= patience) {
                std::cout << "Early stopping triggered after " << (epoch + 1) << " epochs" << std::endl;
                break;
            }
        } else {
            patience_counter = 0;
            lr_patience_counter = 0;
        }
    }
    
    // Print final best metrics
    std::cout << "\n=== Training Completed ===\n"
              << "Best accuracy: " << best_accuracy << "%\n"
              << "Best loss: " << best_loss << "\n"
              << "Final learning rate: " << current_lr << "\n"
              << "======================" << std::endl;
}