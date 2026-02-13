import numpy as np
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate XOR dataset (same as in notebook)
def create_xor_cloud(n_samples=500, margin=0.3):
    """Creates a 2D point cloud with an XOR-like distribution."""
    X_data = np.random.uniform(low=-2, high=2, size=(n_samples, 2))
    
    x_abs = np.abs(X_data[:, 0])
    y_abs = np.abs(X_data[:, 1])
    
    x_in_margin = x_abs < margin
    y_in_margin = y_abs < margin
    
    X_data[x_in_margin, 0] = (x_abs[x_in_margin] + margin) * np.sign(X_data[x_in_margin, 0])
    X_data[y_in_margin, 1] = (y_abs[y_in_margin] + margin) * np.sign(X_data[y_in_margin, 1])
    
    y_data = np.logical_xor(X_data[:, 0] > 0, X_data[:, 1] > 0).astype(int)
    
    return X_data, y_data

# Generate data
np.random.seed(42)
X_data, y_data = create_xor_cloud(n_samples=500, margin=0.3)

# Activation functions and their derivatives
def get_activation_fn(name):
    if name == "Tanh":
        return np.tanh, lambda x: 1 - np.tanh(x)**2
    elif name == "ReLU":
        return lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)
    elif name == "Sigmoid":
        sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))
    elif name == "Leaky ReLU":
        return lambda x: np.where(x > 0, x, 0.01 * x), lambda x: np.where(x > 0, 1.0, 0.01)
    
def softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def forward_pass(x, W1, b1, W2, b2, W3, b3, activation_name):
    """Complete forward pass with all intermediate values"""
    act_fn, _ = get_activation_fn(activation_name)
    
    # Layer 1
    z1 = x @ W1 + b1  # Pre-activation
    a1 = act_fn(z1)    # Post-activation
    
    # Layer 2
    z2 = a1 @ W2 + b2  # Pre-activation
    a2 = act_fn(z2)    # Post-activation
    
    # Output layer
    z3 = a2 @ W3 + b3  # Logits
    probs = softmax(z3)  # Probabilities
    
    return z1, a1, z2, a2, z3, probs

def backward_pass(x, y_true, W1, b1, W2, b2, W3, b3, 
                  z1, a1, z2, a2, z3, probs, activation_name):
    """Complete backward pass - compute all gradients"""
    _, act_derivative = get_activation_fn(activation_name)
    
    # Convert y_true to one-hot
    y_onehot = np.zeros(2)
    y_onehot[y_true] = 1
    
    # Output layer gradients
    # dL/dz3 = probs - y_onehot (derivative of cross-entropy + softmax)
    dL_dz3 = probs - y_onehot
    dL_dW3 = np.outer(a2, dL_dz3)
    dL_db3 = dL_dz3
    
    # Hidden layer 2 gradients
    dL_da2 = dL_dz3 @ W3.T
    dL_dz2 = dL_da2 * act_derivative(z2)
    dL_dW2 = np.outer(a1, dL_dz2)
    dL_db2 = dL_dz2
    
    # Hidden layer 1 gradients
    dL_da1 = dL_dz2 @ W2.T
    dL_dz1 = dL_da1 * act_derivative(z1)
    dL_dW1 = np.outer(x, dL_dz1)
    dL_db1 = dL_dz1
    
    return {
        'dL_dz3': dL_dz3, 'dL_dW3': dL_dW3, 'dL_db3': dL_db3,
        'dL_da2': dL_da2, 'dL_dz2': dL_dz2, 'dL_dW2': dL_dW2, 'dL_db2': dL_db2,
        'dL_da1': dL_da1, 'dL_dz1': dL_dz1, 'dL_dW1': dL_dW1, 'dL_db1': dL_db1
    }

def compute_loss(probs, y_true):
    """Cross-entropy loss"""
    return -np.log(probs[y_true] + 1e-10)

def create_transformation_plots(X, y, W1, b1, W2, b2, W3, b3, activation_name, sample_idx):
    """Create all transformation plots"""
    act_fn, _ = get_activation_fn(activation_name)
    
    # Forward pass for all data
    z1_all = X @ W1 + b1
    a1_all = act_fn(z1_all)
    z2_all = a1_all @ W2 + b2
    a2_all = act_fn(z2_all)
    z3_all = a2_all @ W3 + b3
    probs_all = softmax(z3_all)
    
    # Colors
    colors = ['red' if label == 0 else 'blue' for label in y]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '1. Input Data (Original)',
            '2. Hidden Layer 1 (Pre-activation)',
            '3. Hidden Layer 1 (Post-activation)',
            '4. Hidden Layer 2 (Pre-activation)',
            '5. Hidden Layer 2 (Post-activation)',
            '6. Decision Boundary'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    datasets = [
        (X, 'Input'),
        (z1_all, 'z1'),
        (a1_all, 'a1'),
        (z2_all, 'z2'),
        (a2_all, 'a2')
    ]
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    # Plot transformations
    for (data, name), (row, col) in zip(datasets, positions):
        # Regular points
        for label in [0, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=data[mask, 0],
                    y=data[mask, 1],
                    mode='markers',
                    marker=dict(size=4, color='red' if label == 0 else 'blue', opacity=0.6),
                    showlegend=False,
                    name=f'Class {label}'
                ),
                row=row, col=col
            )
        
        # Highlight selected sample
        fig.add_trace(
            go.Scatter(
                x=[data[sample_idx, 0]],
                y=[data[sample_idx, 1]],
                mode='markers',
                marker=dict(size=15, color='yellow', symbol='star', 
                           line=dict(width=2, color='black')),
                showlegend=False,
                name='Selected'
            ),
            row=row, col=col
        )
    
    # Decision boundary plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    _, _, _, _, _, grid_probs = forward_pass(grid_points, W1, b1, W2, b2, W3, b3, activation_name)
    grid_preds = np.argmax(grid_probs, axis=1).reshape(xx.shape)
    
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=grid_preds,
            colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(0,0,255,0.3)']],
            showscale=False,
            hoverinfo='skip'
        ),
        row=2, col=3
    )
    
    # Add data points on decision boundary
    for label in [0, 1]:
        mask = y == label
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(size=4, color='red' if label == 0 else 'blue', opacity=0.8),
                showlegend=False
            ),
            row=2, col=3
        )
    
    # Highlight selected sample
    fig.add_trace(
        go.Scatter(
            x=[X[sample_idx, 0]],
            y=[X[sample_idx, 1]],
            mode='markers',
            marker=dict(size=15, color='yellow', symbol='star',
                       line=dict(width=2, color='black')),
            showlegend=False
        ),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Neural Network Transformations")
    
    return fig

def format_matrix(matrix, name):
    """Format matrix for display"""
    if matrix.ndim == 1:
        return f"{name}:\n[{', '.join([f'{v:.4f}' for v in matrix])}]"
    else:
        rows = []
        for row in matrix:
            rows.append("  [" + ", ".join([f"{v:.4f}" for v in row]) + "]")
        return f"{name}:\n[" + "\n ".join(rows) + "]"

def update_visualization(sample_idx, activation_fn,
                         w1_00, w1_01, w1_10, w1_11, b1_0, b1_1,
                         w2_00, w2_01, w2_10, w2_11, b2_0, b2_1,
                         w3_00, w3_01, w3_10, w3_11, b3_0, b3_1):
    """Main update function"""
    
    # Construct weight matrices
    W1 = np.array([[w1_00, w1_01], [w1_10, w1_11]])
    b1 = np.array([b1_0, b1_1])
    W2 = np.array([[w2_00, w2_01], [w2_10, w2_11]])
    b2 = np.array([b2_0, b2_1])
    W3 = np.array([[w3_00, w3_01], [w3_10, w3_11]])
    b3 = np.array([b3_0, b3_1])
    
    # Get sample
    x = X_data[sample_idx]
    y_true = y_data[sample_idx]
    
    # Forward pass
    z1, a1, z2, a2, z3, probs = forward_pass(x, W1, b1, W2, b2, W3, b3, activation_fn)
    
    # Compute loss
    loss = compute_loss(probs, y_true)
    
    # Backward pass
    grads = backward_pass(x, y_true, W1, b1, W2, b2, W3, b3,
                         z1, a1, z2, a2, z3, probs, activation_fn)
    
    # Create transformation plots
    fig = create_transformation_plots(X_data, y_data, W1, b1, W2, b2, W3, b3, 
                                     activation_fn, sample_idx)
    
    # Format forward pass info
    y_onehot = np.zeros(2)
    y_onehot[y_true] = 1
    
    forward_info = f"""
## FORWARD PASS (Sample {sample_idx})

**Input:**
x = [{x[0]:.4f}, {x[1]:.4f}]
True Label = {y_true}
One-hot = [{y_onehot[0]:.0f}, {y_onehot[1]:.0f}]

---

**Layer 1:**
z1 = W1·x + b1 = [{z1[0]:.4f}, {z1[1]:.4f}]
a1 = {activation_fn}(z1) = [{a1[0]:.4f}, {a1[1]:.4f}]

**Layer 2:**
z2 = W2·a1 + b2 = [{z2[0]:.4f}, {z2[1]:.4f}]
a2 = {activation_fn}(z2) = [{a2[0]:.4f}, {a2[1]:.4f}]

**Output Layer:**
z3 = W3·a2 + b3 = [{z3[0]:.4f}, {z3[1]:.4f}] (logits)
probs = softmax(z3) = [{probs[0]:.4f}, {probs[1]:.4f}]

---

**Loss Computation:**
Predicted: [{probs[0]:.4f}, {probs[1]:.4f}]
Target:    [{y_onehot[0]:.0f}, {y_onehot[1]:.0f}]
Cross-Entropy Loss = -log(prob_true_class) = {loss:.4f}
"""
    
    # Format backward pass info
    backward_info = f"""
## BACKWARD PASS (Gradients)

**Output Layer:**
∂L/∂z3 (logits) = [{grads['dL_dz3'][0]:.4f}, {grads['dL_dz3'][1]:.4f}]

{format_matrix(grads['dL_dW3'], '∂L/∂W3')}

∂L/∂b3 = [{grads['dL_db3'][0]:.4f}, {grads['dL_db3'][1]:.4f}]

---

**Hidden Layer 2:**
∂L/∂a2 = [{grads['dL_da2'][0]:.4f}, {grads['dL_da2'][1]:.4f}]
∂L/∂z2 (pre-activation) = [{grads['dL_dz2'][0]:.4f}, {grads['dL_dz2'][1]:.4f}]

{format_matrix(grads['dL_dW2'], '∂L/∂W2')}

∂L/∂b2 = [{grads['dL_db2'][0]:.4f}, {grads['dL_db2'][1]:.4f}]

---

**Hidden Layer 1:**
∂L/∂a1 = [{grads['dL_da1'][0]:.4f}, {grads['dL_da1'][1]:.4f}]
∂L/∂z1 (pre-activation) = [{grads['dL_dz1'][0]:.4f}, {grads['dL_dz1'][1]:.4f}]

{format_matrix(grads['dL_dW1'], '∂L/∂W1')}

∂L/∂b1 = [{grads['dL_db1'][0]:.4f}, {grads['dL_db1'][1]:.4f}]

---

**Note:** The gradients show how the loss changes with respect to each parameter.
Larger magnitude = stronger influence on loss.
"""
    
    return fig, forward_info, backward_info

# Initialize with reasonable random weights
np.random.seed(42)
W1_init = np.random.randn(2, 2) * 0.5
b1_init = np.zeros(2)
W2_init = np.random.randn(2, 2) * 0.5
b2_init = np.zeros(2)
W3_init = np.random.randn(2, 2) * 0.5
b3_init = np.zeros(2)

# Create Gradio interface
with gr.Blocks(title="Neural Network Backpropagation Visualizer") as demo:
    gr.Markdown("""
    # 🧠 Neural Network Training Visualizer
    
    **Architecture:** Input(2) → Hidden1(2) → Hidden2(2) → Output(2) with Softmax
    
    This interactive tool shows:
    - **Forward propagation** through each layer
    - **Backward propagation** with all gradients
    - **Representation transformations** at each layer
    - How changing parameters affects the network
    
    **Instructions:**
    1. Select a sample point from the dataset
    2. Choose an activation function
    3. Adjust the weights and biases using sliders
    4. Observe how gradients flow backward through the network
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Control Panel")
            
            sample_slider = gr.Slider(
                minimum=0, maximum=len(X_data)-1, step=1, value=100,
                label="Sample Index"
            )
            
            activation_dropdown = gr.Dropdown(
                choices=["Tanh", "ReLU", "Sigmoid", "Leaky ReLU"],
                value="Tanh",
                label="Activation Function"
            )
            
            gr.Markdown("### Layer 1 Parameters (Input → Hidden1)")
            with gr.Row():
                w1_00 = gr.Slider(-2, 2, W1_init[0,0], step=0.1, label="W1[0,0]")
                w1_01 = gr.Slider(-2, 2, W1_init[0,1], step=0.1, label="W1[0,1]")
            with gr.Row():
                w1_10 = gr.Slider(-2, 2, W1_init[1,0], step=0.1, label="W1[1,0]")
                w1_11 = gr.Slider(-2, 2, W1_init[1,1], step=0.1, label="W1[1,1]")
            with gr.Row():
                b1_0 = gr.Slider(-2, 2, b1_init[0], step=0.1, label="b1[0]")
                b1_1 = gr.Slider(-2, 2, b1_init[1], step=0.1, label="b1[1]")
            
            gr.Markdown("### Layer 2 Parameters (Hidden1 → Hidden2)")
            with gr.Row():
                w2_00 = gr.Slider(-2, 2, W2_init[0,0], step=0.1, label="W2[0,0]")
                w2_01 = gr.Slider(-2, 2, W2_init[0,1], step=0.1, label="W2[0,1]")
            with gr.Row():
                w2_10 = gr.Slider(-2, 2, W2_init[1,0], step=0.1, label="W2[1,0]")
                w2_11 = gr.Slider(-2, 2, W2_init[1,1], step=0.1, label="W2[1,1]")
            with gr.Row():
                b2_0 = gr.Slider(-2, 2, b2_init[0], step=0.1, label="b2[0]")
                b2_1 = gr.Slider(-2, 2, b2_init[1], step=0.1, label="b2[1]")
            
            gr.Markdown("### Output Layer Parameters (Hidden2 → Output)")
            with gr.Row():
                w3_00 = gr.Slider(-2, 2, W3_init[0,0], step=0.1, label="W3[0,0]")
                w3_01 = gr.Slider(-2, 2, W3_init[0,1], step=0.1, label="W3[0,1]")
            with gr.Row():
                w3_10 = gr.Slider(-2, 2, W3_init[1,0], step=0.1, label="W3[1,0]")
                w3_11 = gr.Slider(-2, 2, W3_init[1,1], step=0.1, label="W3[1,1]")
            with gr.Row():
                b3_0 = gr.Slider(-2, 2, b3_init[0], step=0.1, label="b3[0]")
                b3_1 = gr.Slider(-2, 2, b3_init[1], step=0.1, label="b3[1]")
        
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Network Transformations")
            
            with gr.Row():
                with gr.Column():
                    forward_output = gr.Markdown(label="Forward Pass")
                with gr.Column():
                    backward_output = gr.Markdown(label="Backward Pass (Gradients)")
    
    # Connect all inputs to update function
    inputs = [
        sample_slider, activation_dropdown,
        w1_00, w1_01, w1_10, w1_11, b1_0, b1_1,
        w2_00, w2_01, w2_10, w2_11, b2_0, b2_1,
        w3_00, w3_01, w3_10, w3_11, b3_0, b3_1
    ]
    
    outputs = [plot_output, forward_output, backward_output]
    
    # Update on any input change
    for inp in inputs:
        inp.change(fn=update_visualization, inputs=inputs, outputs=outputs)
    
    # Initial load
    demo.load(fn=update_visualization, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")
