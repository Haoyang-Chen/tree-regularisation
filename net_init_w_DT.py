import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_classification
import networkx as nx
from collections import defaultdict, deque


class DTNeuralNetwork(nn.Module):
    def __init__(self, num_features, layer_sizes):
        super(DTNeuralNetwork, self).__init__()
        self.num_features = num_features
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Create layers
        self.layers = nn.ModuleList()

        # First layer (input to first hidden)
        self.layers.append(nn.Linear(num_features, layer_sizes[0], bias=True))

        # Hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i], bias=True))

    def forward(self, x):
        # First layer with sigmoid (logic space transformation)
        x = torch.sigmoid(self.layers[0](x))

        # Intermediate layers with sigmoid
        for i in range(1, len(self.layers) - 1):
            x = torch.relu(self.layers[i](x)*1.5)

        # Output layer with sigmoid
        x = torch.sigmoid(self.layers[-1](x))
        return x


class DTToNNMapper:
    def __init__(self, decision_tree, feature_names=None):
        self.tree = decision_tree.tree_
        self.feature_names = feature_names or [f'x_{i}' for i in range(decision_tree.n_features_in_)]
        self.num_features = decision_tree.n_features_in_
        self.num_classes = decision_tree.n_classes_

        # Step 1: Calculate feature usage count
        self.feature_counts = self._calculate_feature_counts()
        print(f"Feature counts: {self.feature_counts}")

        # Step 3: Calculate depth (needed before layer sizes)
        self.depth = self._calculate_depth()
        print(f"Tree depth: {self.depth}")

        # Step 2: Calculate layer sizes
        self.layer_sizes = self._calculate_layer_sizes()
        print(f"Layer sizes: {self.layer_sizes}")

        # Step 4: Build feature to neuron mapping
        self.feature_neuron_mapping = self._build_feature_mapping()
        print(f"Feature neuron mapping: {self.feature_neuron_mapping}")

        # Create the neural network
        self.nn = DTNeuralNetwork(self.num_features, self.layer_sizes)

        # Step 5 & 6: Initialize the network
        self._initialize_network()

    def _calculate_feature_counts(self):
        """Step 1: Count how many times each feature appears in the tree"""
        feature_counts = defaultdict(int)

        def count_features(node_id):
            if self.tree.children_left[node_id] == self.tree.children_right[node_id]:  # Leaf node
                return

            feature = self.tree.feature[node_id]
            feature_counts[feature] += 1

            # Recursively count in children
            count_features(self.tree.children_left[node_id])
            count_features(self.tree.children_right[node_id])

        count_features(0)  # Start from root
        return dict(feature_counts)

    def _calculate_layer_sizes(self):
        """Step 2: Calculate number of neurons for each layer"""
        # First layer: 2 * sum(n_i) for all features used
        first_layer_size = 2 * sum(self.feature_counts.values())

        # Intermediate layers: same size as first layer
        # Last layer: number of classes
        layer_sizes = [first_layer_size] * self.depth
        layer_sizes[-1] = self.num_classes

        return layer_sizes

    def _calculate_depth(self):
        """Step 3: Calculate depth of the tree (counting leaf level)"""

        def get_depth(node_id):
            if self.tree.children_left[node_id] == self.tree.children_right[node_id]:  # Leaf
                return 1

            left_depth = get_depth(self.tree.children_left[node_id])
            right_depth = get_depth(self.tree.children_right[node_id])
            return max(left_depth, right_depth) + 1

        return get_depth(0)

    def _build_feature_mapping(self):
        """Step 4: Build mapping from features to neuron indices"""
        mapping = {}
        current_idx = 0

        for feature_id in sorted(self.feature_counts.keys()):
            n_i = self.feature_counts[feature_id]
            mapping[feature_id] = (current_idx, current_idx + 2 * n_i - 1)
            current_idx += 2 * n_i

        return mapping

    def _initialize_network(self):
        """Step 5: Initialize first layer and Step 6: Initialize following layers"""
        # Step 5: Initialize first layer
        self._initialize_first_layer()

        # Step 6: Initialize following layers recursively
        self._initialize_following_layers()

    def _initialize_first_layer(self):
        """Initialize first layer with thresholds"""
        with torch.no_grad():
            # Initialize weights and biases to zero
            self.nn.layers[0].weight.fill_(0)
            self.nn.layers[0].bias.fill_(0)

            # Set up threshold neurons for each feature
            node_feature_usage = {}  # Track which neurons correspond to which tree nodes
            neuron_idx = 0

            # Collect nodes in the desired order based on feature mapping
            ordered_nodes = []

            def collect_nodes(node_id):
                if self.tree.children_left[node_id] == self.tree.children_right[node_id]:  # Leaf
                    return
                ordered_nodes.append(node_id)
                collect_nodes(self.tree.children_left[node_id])
                collect_nodes(self.tree.children_right[node_id])

            collect_nodes(0)  # Start from root

            # Sort nodes by feature order
            ordered_nodes.sort(key=lambda node_id: self.tree.feature[node_id])

            # Process nodes in the desired order
            for node_id in ordered_nodes:
                feature = self.tree.feature[node_id]
                threshold = self.tree.threshold[node_id]

                # Store mapping for this specific tree node
                node_feature_usage[node_id] = neuron_idx

                # True neuron (x < threshold)
                self.nn.layers[0].weight[neuron_idx, feature] = -1.0
                self.nn.layers[0].bias[neuron_idx] = threshold

                # False neuron (x >= threshold)
                self.nn.layers[0].weight[neuron_idx + 1, feature] = 1.0
                self.nn.layers[0].bias[neuron_idx + 1] = -threshold

                neuron_idx += 2

            self.node_feature_usage = node_feature_usage

    def _initialize_following_layers(self):
        """Initialize intermediate and output layers"""
        with torch.no_grad():
            # Initialize all layers to zero
            for layer in self.nn.layers[1:]:
                layer.weight.fill_(0)
                layer.bias.fill_(0)

            # Build connections recursively from root
            self._build_connections(0, 0)  # Start from root node, layer 0

    def _build_connections(self, node_id, current_layer):
        """Recursively build connections following the algorithm"""
        if current_layer >= len(self.nn.layers) - 1:  # Reached output layer
            return

        if self.tree.children_left[node_id] == self.tree.children_right[node_id]:  # Leaf node
            # Connect to output layer
            class_label = np.argmax(self.tree.value[node_id][0])
            current_neuron = self.node_feature_usage.get(node_id, 0)

            # Build path to output layer
            for layer_idx in range(current_layer + 1, len(self.nn.layers)):
                if layer_idx == len(self.nn.layers) - 1:  # Output layer
                    self.nn.layers[layer_idx].weight[class_label, current_neuron] = 1.0
                else:  # Intermediate layers
                    if current_neuron < self.nn.layers[layer_idx].weight.shape[1]:
                        self.nn.layers[layer_idx].weight[current_neuron, current_neuron] = 1.0
            return

        # Get current neuron indices
        current_neuron_true = self.node_feature_usage[node_id]
        current_neuron_false = current_neuron_true + 1

        left_child = self.tree.children_left[node_id]
        right_child = self.tree.children_right[node_id]

        # Process left child (true path)
        if self.tree.children_left[left_child] == self.tree.children_right[left_child]:  # Left is leaf
            class_label = np.argmax(self.tree.value[left_child][0])
            # Build direct connection to output
            for layer_idx in range(current_layer + 1, len(self.nn.layers)):
                if layer_idx == len(self.nn.layers) - 1:  # Output layer
                    self.nn.layers[layer_idx].weight[class_label, current_neuron_true] = 1.0
                else:  # Intermediate layers - create identity path
                    if (current_neuron_true < self.nn.layers[layer_idx].weight.shape[0] and
                            current_neuron_true < self.nn.layers[layer_idx].weight.shape[1]):
                        self.nn.layers[layer_idx].weight[current_neuron_true, current_neuron_true] = 1.0
        else:  # Left is condition node
            left_neuron_true = self.node_feature_usage[left_child]
            left_neuron_false = left_neuron_true + 1
            # Connect current true neuron to left child's neurons with AND logic (weight=1, bias=-1)
            if current_layer + 1 < len(self.nn.layers):
                self.nn.layers[current_layer + 1].weight[left_neuron_true, current_neuron_true] = 1.0
                self.nn.layers[current_layer + 1].bias[left_neuron_true] = -1.0
                self.nn.layers[current_layer + 1].weight[left_neuron_false, current_neuron_true] = 1.0
                self.nn.layers[current_layer + 1].bias[left_neuron_false] = -1.0
                for layer_idx in range(1, current_layer +2):
                    self.nn.layers[layer_idx].weight[left_neuron_true, left_neuron_true] = 1.0
                    self.nn.layers[layer_idx].weight[left_neuron_false, left_neuron_false] = 1.0
            self._build_connections(left_child, current_layer + 1)

        # Process right child (false path)
        if self.tree.children_left[right_child] == self.tree.children_right[right_child]:  # Right is leaf
            class_label = np.argmax(self.tree.value[right_child][0])
            # Build direct connection to output
            for layer_idx in range(current_layer + 1, len(self.nn.layers)):
                if layer_idx == len(self.nn.layers) - 1:  # Output layer
                    self.nn.layers[layer_idx].weight[class_label, current_neuron_false] = 1.0
                else:  # Intermediate layers - create identity path
                    if (current_neuron_false < self.nn.layers[layer_idx].weight.shape[0] and
                            current_neuron_false < self.nn.layers[layer_idx].weight.shape[1]):
                        self.nn.layers[layer_idx].weight[current_neuron_false, current_neuron_false] = 1.0
        else:  # Right is condition node
            right_neuron_true = self.node_feature_usage[right_child]
            right_neuron_false = right_neuron_true + 1
            # Connect current false neuron to right child's neurons
            if current_layer + 1 < len(self.nn.layers):
                self.nn.layers[current_layer + 1].weight[right_neuron_true, current_neuron_false] = 1.0
                self.nn.layers[current_layer + 1].bias[right_neuron_true] = -1.0
                self.nn.layers[current_layer + 1].weight[right_neuron_false, current_neuron_false] = 1.0
                self.nn.layers[current_layer + 1].bias[right_neuron_false] = -1.0
                for layer_idx in range(1, current_layer + 2):
                    self.nn.layers[layer_idx].weight[right_neuron_true, right_neuron_true] = 1.0
                    self.nn.layers[layer_idx].weight[right_neuron_false, right_neuron_false] = 1.0
            self._build_connections(right_child, current_layer + 1)


def create_sample_tree():
    """Create a sample decision tree for testing"""
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0,
                               n_informative=4, n_classes=2, random_state=42)

    # Create and train decision tree
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    return dt, X, y


def visualize_tree_structure(dt):
    """Visualize the decision tree structure"""
    tree_text = export_text(dt, feature_names=[f'x_{i}' for i in range(dt.n_features_in_)])
    print("Decision Tree Structure:")
    print(tree_text)


def test_mapping():
    """Test the DT-NN mapping algorithm"""
    print("Creating sample decision tree...")
    dt, X, y = create_sample_tree()

    print("\nDecision Tree Structure:")
    visualize_tree_structure(dt)

    print("\nMapping Decision Tree to Neural Network...")
    mapper = DTToNNMapper(dt)

    print(f"\nNeural Network Architecture:")
    print(f"Input features: {mapper.num_features}")
    print(f"Layer sizes: {mapper.layer_sizes}")
    print(f"Total layers: {len(mapper.layer_sizes)}")

    # Test with some sample data
    print("\nTesting predictions...")
    test_samples = X[:5]  # First 5 samples
    print(test_samples)

    # Get DT predictions
    dt_predictions = dt.predict(test_samples)
    dt_probabilities = dt.predict_proba(test_samples)

    # Get NN predictions
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_samples)
        nn_outputs = mapper.nn(test_tensor)
        # For binary classification, take argmax to get class prediction
        nn_predictions = torch.argmax(nn_outputs, dim=1).numpy()

    print("\nComparison of predictions:")
    print("Sample | DT Pred | DT Prob | NN Output (Class 0, Class 1) | NN Pred")
    print("-" * 70)
    for i in range(len(test_samples)):
        nn_out_0 = nn_outputs[i][0].item()
        nn_out_1 = nn_outputs[i][1].item()
        print(
            f"{i:6d} | {dt_predictions[i]:7d} | {dt_probabilities[i][1]:7.3f} | ({nn_out_0:5.3f}, {nn_out_1:5.3f}) | {nn_predictions[i]:7d}")

    return mapper


def visualize_network_architecture(mapper):
    """Create a detailed visualization of the network architecture with labels and weights"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))

    # Left plot: Decision Tree Structure
    ax1.set_title('Decision Tree Structure', fontsize=14, fontweight='bold')

    # Visualize tree nodes and thresholds
    def draw_tree_node(node_id, x, y, width=1.0, depth=0):
        if mapper.tree.children_left[node_id] == mapper.tree.children_right[node_id]:  # Leaf
            # Draw leaf
            class_label = np.argmax(mapper.tree.value[node_id][0])
            ax1.add_patch(plt.Circle((x, y), 0.15, color='lightgreen', ec='black', linewidth=2))
            ax1.text(x, y, f'Class\n{class_label}', ha='center', va='center', fontweight='bold', fontsize=9)
            return

        # Draw internal node
        feature = mapper.tree.feature[node_id]
        threshold = mapper.tree.threshold[node_id]
        ax1.add_patch(plt.Circle((x, y), 0.2, color='lightblue', ec='black', linewidth=2))
        ax1.text(x, y + 0.05, f'x_{feature}', ha='center', va='center', fontweight='bold', fontsize=8)
        ax1.text(x, y - 0.05, f'< {threshold:.2f}', ha='center', va='center', fontsize=7)

        # Draw children
        child_width = width / 2
        left_child = mapper.tree.children_left[node_id]
        right_child = mapper.tree.children_right[node_id]

        # Left child (True)
        left_x = x - width / 3
        left_y = y - 1.2
        ax1.plot([x, left_x], [y - 0.2, left_y + 0.2], 'k-', linewidth=2)
        ax1.text((x + left_x) / 2 - 0.15, (y + left_y) / 2, 'True', fontsize=9, color='green', fontweight='bold')
        draw_tree_node(left_child, left_x, left_y, child_width, depth + 1)

        # Right child (False)
        right_x = x + width / 3
        right_y = y - 1.2
        ax1.plot([x, right_x], [y - 0.2, right_y + 0.2], 'k-', linewidth=2)
        ax1.text((x + right_x) / 2 + 0.15, (y + right_y) / 2, 'False', fontsize=9, color='red', fontweight='bold')
        draw_tree_node(right_child, right_x, right_y, child_width, depth + 1)

    draw_tree_node(0, 0, 0, 2.5)  # Start from root
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-5, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Right plot: Neural Network Architecture
    ax2.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')

    # Calculate proper spacing and positioning
    total_layers = len(mapper.layer_sizes) + 1  # +1 for input layer
    layer_spacing = 4.0
    max_neurons = max(mapper.layer_sizes + [mapper.num_features])
    neuron_spacing = 1.2

    # Store layer information for organized display
    layer_info = []

    # Input layer
    input_neurons = mapper.num_features
    input_y_positions = np.linspace(-(input_neurons - 1) * neuron_spacing / 2, (input_neurons - 1) * neuron_spacing / 2,
                                    input_neurons)[::-1]  # Reverse the order for top-to-bottom drawing
    input_x = 0
    layer_info.append({
        'x': input_x,
        'y_positions': input_y_positions,
        'size': input_neurons,
        'type': 'input',
        'color': '#FF6B6B'  # Red
    })

    # Hidden and output layers
    colors = ['#FFA726', '#FFEE58', '#66BB6A', '#42A5F5']  # Orange, Yellow, Green, Blue
    for i, layer_size in enumerate(mapper.layer_sizes):
        x_pos = (i + 1) * layer_spacing
        y_positions = np.linspace(-(layer_size - 1) * neuron_spacing / 2, (layer_size - 1) * neuron_spacing / 2,
                                  layer_size)[::-1]  # Reverse the order for top-to-bottom drawing
        layer_type = 'output' if i == len(mapper.layer_sizes) - 1 else 'hidden'

        layer_info.append({
            'x': x_pos,
            'y_positions': y_positions,
            'size': layer_size,
            'type': layer_type,
            'color': colors[i % len(colors)],
            'layer_idx': i
        })

    # Draw neurons with clear, organized labels
    for layer_idx, layer in enumerate(layer_info):
        x_pos = layer['x']
        y_positions = layer['y_positions']

        for neuron_idx in range(layer['size']):
            y_pos = y_positions[neuron_idx]

            # Draw neuron circle (larger for better visibility)
            circle = plt.Circle((x_pos, y_pos), 0.4, color=layer['color'],
                                ec='black', linewidth=2, alpha=0.9)
            ax2.add_patch(circle)

            if layer['type'] == 'input':
                # Clear input feature labels
                ax2.text(x_pos, y_pos, f'x{neuron_idx}', ha='center', va='center',
                         fontweight='bold', fontsize=12, color='white')
            else:
                # Get bias for this neuron from the correct layer
                bias = mapper.nn.layers[layer['layer_idx']].bias[neuron_idx].item()
                # Clear neuron labels with bias below
                if layer['type'] == 'output':
                    ax2.text(x_pos, y_pos + 0.15, f'out{neuron_idx}',
                             ha='center', va='center', fontweight='bold', fontsize=10)
                else:
                    ax2.text(x_pos, y_pos + 0.15, f'h{layer["layer_idx"]}_{neuron_idx}',
                             ha='center', va='center', fontweight='bold', fontsize=10)
                ax2.text(x_pos, y_pos - 0.15, f'{bias:.2f}',
                         ha='center', va='center', fontsize=9, style='italic', color='darkblue')

        # Layer titles with better positioning
        title_y = max(y_positions) + 1.0
        if layer['type'] == 'input':
            title = 'Input\nLayer'
        elif layer['type'] == 'output':
            title = 'Output\nLayer'
        else:
            title = f'Hidden\nLayer {layer["layer_idx"] + 1}'

        ax2.text(x_pos, title_y, title, ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

    # Draw connections with selective weight display
    for layer_idx in range(len(mapper.nn.layers)):
        weights = mapper.nn.layers[layer_idx].weight.detach().numpy()
        from_layer = layer_info[layer_idx]
        to_layer = layer_info[layer_idx + 1]

        # # Debug info
        # print(f"Drawing connections from layer {layer_idx} to layer {layer_idx + 1}")
        # print(f"Weight matrix shape: {weights.shape}")
        # print(f"From layer size: {from_layer['size']}, To layer size: {to_layer['size']}")

        # Only show weights for the most significant connections to avoid clutter
        weight_threshold = np.percentile(np.abs(weights), 70)  # Top 30% of weights

        connections_drawn = 0
        for i in range(weights.shape[0]):  # To neurons (output neurons of this layer)
            for j in range(weights.shape[1]):  # From neurons (input neurons to this layer)
                weight_val = weights[i, j]

                if abs(weight_val) > weight_threshold:
                    from_x, from_y = from_layer['x'], from_layer['y_positions'][j]
                    to_x, to_y = to_layer['x'], to_layer['y_positions'][i]

                    # Line styling based on weight
                    alpha = min(abs(weight_val) * 1.5, 0.8)
                    linewidth = max(abs(weight_val) * 3, 1.0)
                    color = '#D32F2F' if weight_val < 0 else '#1976D2'  # Red/Blue

                    # Draw connection
                    ax2.plot([from_x, to_x], [from_y, to_y],
                             color=color, alpha=alpha, linewidth=linewidth)
                    connections_drawn += 1

                    # Add weight label only for very significant weights (avoid overcrowding)
                    if abs(weight_val) > np.percentile(np.abs(weights), 85):  # Top 15%
                        mid_x = (from_x + to_x) / 2 + 0.2  # Slight offset to avoid overlap
                        mid_y = (from_y + to_y) / 2
                        ax2.text(mid_x, mid_y, f'{weight_val:.2f}', ha='center', va='center',
                                 fontsize=8, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                           edgecolor='gray', alpha=0.9))

        print(f"Drew {connections_drawn} connections for layer {layer_idx}")
        print("---")

    # Clean legend
    legend_x = (len(mapper.layer_sizes)) * layer_spacing + 1
    legend_y = max([max(layer['y_positions']) for layer in layer_info]) - 1
    ax2.text(legend_x, legend_y,
             'Connections:\n• Blue = Positive weight\n• Red = Negative weight\n• Thickness ∝ |weight|\n\nNeurons:\n• Numbers below = bias values',
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgray', alpha=0.9))

    # Set clean axis limits
    ax2.set_xlim(-1, (len(mapper.layer_sizes)) * layer_spacing + 3)
    ax2.set_ylim(-max_neurons * neuron_spacing / 2 - 1, max_neurons * neuron_spacing / 2 + 2)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_network_weights(mapper):
    """Visualize the network weights in detail"""
    num_layers = len(mapper.nn.layers)
    fig, axes = plt.subplots(2, num_layers, figsize=(5 * num_layers, 10))
    if num_layers == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle('Neural Network Weights and Biases Visualization', fontsize=16)

    for i, layer in enumerate(mapper.nn.layers):
        weights = layer.weight.detach().numpy()
        biases = layer.bias.detach().numpy()

        # Plot weights
        im = axes[0, i].imshow(weights, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
        axes[0, i].set_title(f'Layer {i} Weights')
        axes[0, i].set_xlabel('Input Neurons' if i == 0 else 'Previous Layer')
        axes[0, i].set_ylabel('Output Neurons')
        plt.colorbar(im, ax=axes[0, i])

        # Add text annotations for non-zero weights
        for row in range(weights.shape[0]):
            for col in range(weights.shape[1]):
                if abs(weights[row, col]) > 0.1:
                    axes[0, i].text(col, row, f'{weights[row, col]:.1f}',
                                    ha='center', va='center', fontsize=6,
                                    color='white' if abs(weights[row, col]) > 1 else 'black')

        # Plot biases
        axes[1, i].bar(range(len(biases)), biases, color='skyblue', edgecolor='black')
        axes[1, i].set_title(f'Layer {i} Biases')
        axes[1, i].set_xlabel('Neuron Index')
        axes[1, i].set_ylabel('Bias Value')
        axes[1, i].grid(True, alpha=0.3)

        # Add value labels on bars
        for j, bias in enumerate(biases):
            if abs(bias) > 0.1:
                axes[1, i].text(j, bias + 0.1 * np.sign(bias), f'{bias:.1f}',
                                ha='center', va='bottom' if bias > 0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the test
    mapper = test_mapping()

    # Visualize the network architecture
    print("\nVisualizing network architecture...")
    visualize_network_architecture(mapper)

    # Visualize the network weights in detail
    print("\nVisualizing network weights...")
    visualize_network_weights(mapper)

    # Additional verification: test the logical equivalence
    print("\nVerifying logical equivalence...")
    dt, X, y = create_sample_tree()

    # Test on a larger sample
    test_samples = X[:20]
    dt_predictions = dt.predict(test_samples)

    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_samples)
        nn_outputs = mapper.nn(test_tensor)
        nn_predictions = torch.argmax(nn_outputs, dim=1).numpy()

    accuracy = np.mean(dt_predictions == nn_predictions)
    print(f"Accuracy of NN vs DT: {accuracy:.3f}")

    if accuracy == 1.0:
        print("✅ Perfect mapping achieved! Neural network is logically equivalent to decision tree.")
    else:
        print("⚠️  Mapping needs adjustment. Some predictions don't match.")

        # Show mismatches
        mismatches = np.where(dt_predictions != nn_predictions)[0]
        if len(mismatches) > 0:
            print(f"Mismatches at indices: {mismatches}")
            for idx in mismatches[:5]:  # Show first 5 mismatches
                nn_out_0 = nn_outputs[idx][0].item()
                nn_out_1 = nn_outputs[idx][1].item()
                print(
                    f"Sample {idx}: DT={dt_predictions[idx]}, NN={nn_predictions[idx]}, NN_output=({nn_out_0:.3f}, {nn_out_1:.3f})")