# Neural Network Backpropagation Visualizer

An interactive educational tool to understand how neural networks work, including forward propagation, backpropagation, and gradient flow.

## Features

### 🎯 What This Tool Shows

1. **Complete Forward Pass**
   - Input values for selected sample
   - Pre-activation values (z) at each layer
   - Post-activation values (a) at each layer
   - Final softmax probabilities
   - Cross-entropy loss computation

2. **Complete Backward Pass (Backpropagation)**
   - Gradients at output layer (∂L/∂z3)
   - Gradients for all weights and biases (∂L/∂W, ∂L/∂b)
   - Gradient flow through activations (∂L/∂a, ∂L/∂z)
   - Shows how gradients propagate from output back to input

3. **Representation Space Transformations**
   - Original XOR data distribution
   - Transformation after Hidden Layer 1 (pre and post activation)
   - Transformation after Hidden Layer 2 (pre and post activation)
   - Final decision boundary

4. **Interactive Parameter Control**
   - 15 sliders to manually adjust all network parameters
   - Real-time updates showing effect on predictions and gradients
   - Choose between activation functions: Tanh, ReLU, Sigmoid, Leaky ReLU

## Network Architecture

```
Input (2D)
    ↓
Hidden Layer 1: 2 neurons + Activation
    ↓
Hidden Layer 2: 2 neurons + Activation
    ↓
Output Layer: 2 neurons + Softmax
    ↓
Cross-Entropy Loss
```

**Why 2 neurons in hidden layers?**
Having exactly 2 neurons in each hidden layer allows us to visualize the transformed representation space in 2D plots, showing how the network learns to separate the XOR pattern.

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install gradio plotly numpy
```

## Usage

1. Run the application:
```bash
python nn_visualizer.py
```

2. Open your browser to `http://localhost:7860`

3. Interact with the tool:
   - **Select a sample**: Use the "Sample Index" slider to choose a point from the dataset
   - **Choose activation**: Select Tanh, ReLU, Sigmoid, or Leaky ReLU
   - **Adjust parameters**: Move the weight and bias sliders to see effects
   - **Observe**:
     - How the decision boundary changes
     - How representation spaces transform
     - How gradients flow backward
     - How loss changes with different parameters

## Understanding the Visualizations

### Forward Pass Section
Shows the complete computation from input to output:
- Each layer's pre-activation (z = W·input + b)
- Each layer's post-activation (a = activation(z))
- Final probabilities and loss

### Backward Pass Section
Shows how gradients propagate backward:
- **∂L/∂z**: Gradient with respect to pre-activation values
- **∂L/∂a**: Gradient with respect to post-activation values
- **∂L/∂W**: Gradient with respect to weights (used for updates)
- **∂L/∂b**: Gradient with respect to biases (used for updates)

**Key insight**: Larger magnitude gradients indicate parameters that have more influence on the loss.

### Transformation Plots
Shows how the network transforms the input space:
1. **Original data**: XOR pattern (not linearly separable)
2. **After Layer 1**: First transformation
3. **After Layer 2**: Second transformation
4. **Decision boundary**: Final classification

Watch how the network "untangles" the XOR pattern through successive transformations!

## Educational Use Cases

### For Students
- Understand what happens at each neuron
- See backpropagation in action with real gradients
- Visualize how activation functions affect gradients (e.g., "dying ReLU")
- Explore how changing parameters affects predictions

### For Instructors
- Demonstrate forward and backward propagation step-by-step
- Show the chain rule in action
- Compare different activation functions
- Illustrate representation learning

## Dataset

The tool uses an XOR-like dataset with 500 points:
- Class 0 (red): Points in quadrants where x·y < 0
- Class 1 (blue): Points in quadrants where x·y > 0
- This pattern is **not linearly separable**, requiring hidden layers

## Tips for Exploration

1. **Start with default parameters** to see a reasonable network
2. **Try different activations**:
   - Tanh: Smooth gradients, centered at zero
   - ReLU: Sparse activations, watch for "dead" neurons
   - Sigmoid: Output range [0,1], can have vanishing gradients
   - Leaky ReLU: Like ReLU but prevents completely dead neurons

3. **Experiment with parameters**:
   - Make weights very large/small to see saturation effects
   - Set some weights to zero to see network simplification
   - Try negative vs positive weights

4. **Observe gradient magnitudes**:
   - Small gradients → slow learning (vanishing gradient)
   - Large gradients → unstable learning (exploding gradient)

## Technical Details

- **Loss Function**: Cross-Entropy with Softmax
- **Dataset**: 500 samples, XOR distribution
- **Gradient Computation**: Full analytical backpropagation
- **Visualization**: Plotly for interactive plots

## Troubleshooting

**Port already in use?**
Change the port in the last line of `nn_visualizer.py`:
```python
demo.launch(server_port=7861)  # Use different port
```

**Plots not updating?**
Refresh the page or adjust any slider to trigger update.

## License

This is an educational tool. Feel free to use and modify for teaching purposes.

## Credits

Based on neural network fundamentals and inspired by similar visualization tools in the ML education community.
