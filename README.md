# CUDA Neural Network (cuda-net)

An implementation and acceleration of a fully connected Neural Network (MLP) on GPU using CUDA C++, developed as a final project for the High Performance Computing course at the University of Calabria.

## Project Overview

This project implements a multi-layer perceptron (MLP) from scratch using direct CUDA programming. By bypassing high-level deep learning frameworks, it achieves superior performance through manual optimization of matrix operations, activation functions, and gradient descent algorithms.

### Key Features

- **Massive Parallelism:** All critical operations (forward pass, backpropagation, weight updates) are implemented as custom CUDA kernels.
    
- **Modular Design:** A `Layer`-based architecture allows for dynamic network configuration.
    
- **Advanced Optimizations:**
    
    - **He Initialization:** Weights initialized based on input size to ensure stable convergence.
        
    - **Gradient Clipping:** Prevents exploding gradients by capping the norm at 1.0.
        
    - **L2 Regularization:** Reduces overfitting by penalizing large weights.
        
    - **Early Stopping:** Automatically halts training when performance on the test set plateaus.
        
    - **Learning Rate Scheduling:** Dynamically reduces the learning rate to fine-tune accuracy.
        
- **Data Augmentation:** Real-time transformation of training data including random rotation, noise addition, and horizontal flipping.
    

## Technical Architecture

### Custom CUDA Kernels

The "engine" of the network consists of several specialized kernels:

- **Matrix Multiplication:** Three variations (`matmul_a_b`, `matmul_a_bt`, `matmul_at_b`) optimized for standard passes and transposed gradient calculations.
    
- **Activations:** `ReLU`, `Leaky ReLU`, `Sigmoid`, and `Softmax`.
    
- **Gradient Computation:** Specific kernels for cross-entropy loss gradients and atomic weight updates.
    

### Classes

- `DataLoader`: Handles binary file parsing and on-the-fly data augmentation.
    
- `Layer`: Manages device memory for weights, biases, and activations for a single layer.
    
- `NeuralNetwork`: Coordinates the full lifecycle of the model (Forward, Backward, Train, Evaluate).
    

## Performance Comparison (MNIST)

The implementation was benchmarked against an equivalent architecture in **PyTorch** using a 784 → 512 → 128 → 10 configuration.

|   |   |   |
|---|---|---|
|**Metric**|**CUDA C++ (This Project)**|**PyTorch (CUDA Backend)**|
|**Total Training Time (20 Epochs)**|**63.32 seconds**|241.54 seconds|
|**Final Test Accuracy**|**97.00%**|94.01%|
|**Speedup**|**~3.8x**|1.0x|

**Insights:**

- **Convergence:** The CUDA implementation reaches >90% accuracy within the first few epochs.
    
- **Efficiency:** Manual memory management and kernel tuning allow for significantly lower overhead compared to the PyTorch abstraction layer.
    

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
    
- CUDA Toolkit (NVCC)
    
- MNIST dataset converted to binary format (`.bin`)
    

### Required Data Files

The application expects the following files in the working directory:

- `X_train.bin` / `y_train.bin`: Training images and labels.
    
- `X_test.bin` / `y_test.bin`: Test images and labels.
    

### Compilation

Use the following command to compile the project:

```
nvcc -o cuda-net main.cu neural_network.cu layer.cu kernels.cu data_loader.cu -lcudart -O2
```

## Configuration

Hyperparameters are defined in `utils.cuh`:

- `BATCH_SIZE`: Default 32
    
- `LEARNING_RATE`: Default 0.001
    
- `HIDDEN_LAYERS`: Configurable sizes (default 512, 128)
    

**Author:** Alessandro La Cava

---
Developed as an educational project at the Università della Calabria.
