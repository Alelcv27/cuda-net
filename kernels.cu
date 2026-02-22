#include "kernels.cuh"

/**
 * Performs matrix multiplication C = A * B
 * @param A Input matrix of shape (m x n)
 * @param B Input matrix of shape (n x k)
 * @param C Output matrix of shape (m x k)
 * @param m Number of rows in A
 * @param n Number of columns in A / rows in B
 * @param k Number of columns in B
 */

__global__ void matmul_a_b_kernel(float* A, float* B, float* C, int m, int n, int k) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if (row < m && col < k) {
       float sum = 0.0f;
       for (int i = 0; i < n; ++i) {
           sum += A[row * n + i] * B[i * k + col];
       }
       C[row * k + col] = sum;
   }
}

/**
 * Performs matrix multiplication C = A * B^T (B transposed)
 * @param A Input matrix of shape (m x n)
 * @param B Input matrix of shape (k x n) - will be transposed
 * @param C Output matrix of shape (m x k)
 * @param m Number of rows in A
 * @param n Number of columns in A / columns in B
 * @param k Number of rows in B
 */

__global__ void matmul_a_bt_kernel(float* A, float* B, float* C, int m, int n, int k) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if (row < m && col < k) {
       float sum = 0.0f;
       for (int i = 0; i < n; ++i) {
           sum += A[row * n + i] * B[col * n + i];
       }
       C[row * k + col] = sum;
   }
}

/**
 * Performs matrix multiplication C = A^T * B (A transposed)
 * @param A Input matrix of shape (m x n) - will be transposed
 * @param B Input matrix of shape (m x k)
 * @param C Output matrix of shape (n x k)
 * @param m Number of rows in A / rows in B
 * @param n Number of columns in A / rows in output C
 * @param k Number of columns in B / columns in output C
 */

__global__ void matmul_at_b_kernel(float* A, float* B, float* C, int m, int n, int k) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if (row < n && col < k) {
       float sum = 0.0f;
       for (int i = 0; i < m; ++i) {
           sum += A[i * n + row] * B[i * k + col];
       }
       C[row * k + col] = sum;
   }
}

/**
 * Applies ReLU activation function element-wise: f(x) = max(0, x)
 * @param x Input/output tensor
 * @param size Number of elements in the tensor
 */

__global__ void relu_kernel(float* x, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       x[idx] = fmaxf(0.0f, x[idx]);
   }
}

/**
 * Applies sigmoid activation function element-wise: f(x) = 1/(1+e^(-x))
 * @param x Input/output tensor
 * @param size Number of elements in the tensor
 */

__global__ void sigmoid_kernel(float* x, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       x[idx] = 1.0f / (1.0f + expf(-x[idx]));
   }
}

/**
 * Computes the derivative of sigmoid activation function
 * @param input Input tensor (pre-activation values, already passed through sigmoid)
 * @param output Output tensor for sigmoid derivatives
 * @param size Number of elements in the tensor
 */

__global__ void dsigmoid_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * (1.0f - input[idx]);
    }
}

/**
 * Adds bias to each element in a batch of vectors
 * @param x Input/output tensor of shape (batch_size x size)
 * @param bias Bias vector of shape (size)
 * @param batch_size Number of samples in the batch
 * @param size Size of each vector
 */

__global__ void bias_add_kernel(float* x, float* bias, int batch_size, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int b = idx / size;
   int i = idx % size;

   if (b < batch_size && i < size) {
       x[idx] += bias[i];
   }
}

/**
 * Applies softmax function to each vector in a batch
 * Softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 * @param x Input/output tensor of shape (batch_size x size)
 * @param batch_size Number of samples in the batch
 * @param size Size of each vector
 */

__global__ void softmax_kernel(float* x, int batch_size, int size) {
   int b = blockIdx.x;
   if (b < batch_size) {
       float max_val = x[b * size];
       for (int i = 1; i < size; ++i) {
           max_val = fmaxf(max_val, x[b * size + i]);
	   } // Find the maximum value in the vector for numerical stability (to avoid working with large numbers)

       float sum = 0.0f;
       for (int i = 0; i < size; ++i) {
           x[b * size + i] = expf(x[b * size + i] - max_val);
           sum += x[b * size + i];
       }

       for (int i = 0; i < size; ++i) {
           x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
       }
   }
}

/**
 * Initializes gradient tensor with zeros
 * @param grad Gradient tensor to be zeroed
 * @param size Number of elements in the tensor
 */

__global__ void zero_grad_kernel(float* grad, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       grad[idx] = 0.0f;
   }
}

/**
 * Computes output gradients for cross-entropy loss with fixed OUTPUT_SIZE
 * @param grad_output Output gradient tensor
 * @param output Network output tensor (after softmax)
 * @param labels Ground truth labels
 * @param batch_size Number of samples in the batch
 */

__global__ void compute_output_gradients_kernel(float* grad_output, float* output, int* labels, int batch_size) {
   int b = blockIdx.x * blockDim.x + threadIdx.x;
   if (b < batch_size) {
       for (int i = 0; i < OUTPUT_SIZE; ++i) {
           grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
       }
       grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
   }
}

/**
 * Computes output gradients for cross-entropy loss with variable output size
 * @param gradients Output gradient tensor
 * @param output Network output tensor (after softmax)
 * @param labels Ground truth labels
 * @param batch_size Number of samples in the batch
 * @param output_size Size of output vectors
 */

__global__ void compute_output_gradients_kernel(float* gradients, float* output, int* labels, 
                                               int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Find the label for this sample
    int label = labels[idx];
    
    // Copy the outputs to gradients
    for (int i = 0; i < output_size; i++) {
        gradients[idx * output_size + i] = output[idx * output_size + i];
    }
    
    // Subtract 1 from the correct class (dL/dz = softmax - one_hot)
    if (label >= 0 && label < output_size) {
        gradients[idx * output_size + label] -= 1.0f;
    }
}

/**
 * Updates weight and bias gradients for a layer
 * @param grad_weights Weight gradients tensor
 * @param grad_bias Bias gradients tensor
 * @param grad_layer Curret layer gradients tensor
 * @param prev_layer Previous layer activations
 * @param batch_size Number of samples in the batch
 * @param prev_size Size of previous layer
 * @param curr_size Size of current layer
 */

__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, 
    float* prev_layer, int batch_size, int prev_size, int curr_size) {
   int i = blockIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < curr_size && j < prev_size) {
       float grad_w_sum = 0.0f;
       for (int b = 0; b < batch_size; ++b) {
           grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
       }
       atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);

       if (j == 0) {
           float grad_b_sum = 0.0f;
           for (int b = 0; b < batch_size; ++b) {
               grad_b_sum += grad_layer[b * curr_size + i];
           }
           atomicAdd(&grad_bias[i], grad_b_sum);
       }
   }
}

/**
 * Computes the derivative of ReLU activation function
 * @param x Input tensor (pre-activation values)
 * @param d_ReLU_out Output tensor for ReLU derivatives
 * @param size Number of elements in the tensor
 */

__global__ void drelu_kernel(float* x, float* d_ReLU_out, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       d_ReLU_out[idx] = x[idx] > 0.0f ? 1.0f : 0.0f;
   }
}

/**
 * Applies LeakyReLU activation function element-wise: f(x) = max(0.01x, x)
 * @param data Input/output tensor
 * @param size Number of elements in the tensor
 */

__global__ void leaky_relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0 ? data[idx] : 0.01f * data[idx]; // 0.01 is the standard alpha value
    }
}

/**
 * Computes the derivative of LeakyReLU activation function
 * @param input Input tensor (pre-activation values)
 * @param output Output tensor for LeakyReLU derivatives
 * @param size Number of elements in the tensor
 */

__global__ void dleaky_relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? 1.0f : 0.01f; // 0.01 is the standard alpha value
    }
}

/**
 * Element-wise multiplication of two gradient tensors
 * @param grad1 First gradient tensor (input/output)
 * @param grad2 Second gradient tensor
 * @param size Number of elements in the tensors
 */

__global__ void multiply_gradients_kernel(float* grad1, float* grad2, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       grad1[idx] *= grad2[idx];
   }
}

/**
 * Updates weights using fixed learning rate (LEARNING_RATE)
 * @param weights Weights tensor to update
 * @param grad_weights Gradient of weights
 * @param size Number of elements in the tensors
 */

__global__ void update_weights_kernel(float* weights, float* grad_weights, int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size) {
       weights[idx] -= LEARNING_RATE * grad_weights[idx];
   }
}

/**
 * Updates weights using specified learning rate
 * @param weights Weights tensor to update
 * @param gradients Gradient of weights
 * @param size Number of elements in the tensors
 * @param learning_rate Learning rate for the update
 */

__global__ void update_weights_kernel(float* weights, float* gradients, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

/**
 * CUDA kernel for updating weights with L2 regularization
 */

__global__ void update_weights_l2_kernel(float* weights, float* gradients, int size, float learning_rate, float l2_lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * (gradients[idx] + l2_lambda * weights[idx]);
    }
}

/**
 * CUDA kernel to clip gradients to a maximum L2 norm
 */
__global__ void clip_gradients_kernel(float* gradients, int size, float threshold) {
    // Calculate L2 norm of gradients
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += gradients[i] * gradients[i];
    }
    norm = sqrtf(norm);

    // Only clip if norm exceeds threshold
    if (norm > threshold) {
        float scale = threshold / norm;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            gradients[idx] *= scale;
        }
    }
}