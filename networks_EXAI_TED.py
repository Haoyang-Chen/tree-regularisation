import warnings
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from zss import simple_distance, Node


np.random.seed(5555)
torch.random.manual_seed(5255)

warnings.filterwarnings('ignore')

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'mps'

class TreeNet(nn.Module):
    def __init__(self, input_dim, min_samples_leaf=1):
        super(TreeNet, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.min_samples_leaf = min_samples_leaf

        self.random_seeds = np.random.randint(1, 100, 10)

        self.original_tree = """2:2
        2:1
        2:0
        0:-2
        0:-2
        2:0
        0:-2
        0:-2
        2:2
        2:2
        0:-2
        0:-2
        2:0
        0:-2
        0:-2"""
        # |--- feature_2 <= 5095.50
        # |   |--- feature_1 <= 12.50
        # |   |   |--- feature_0 <= 33.50
        # |   |   |   |--- class: 0
        # |   |   |--- feature_0 >  33.50
        # |   |   |   |--- class: 0
        # |   |--- feature_1 >  12.50
        # |   |   |--- feature_0 <= 30.50
        # |   |   |   |--- class: 0
        # |   |   |--- feature_0 >  30.50
        # |   |   |   |--- class: 1
        # |--- feature_2 >  5095.50
        # |   |--- feature_2 <= 7073.50
        # |   |   |--- feature_2 <= 5316.50
        # |   |   |   |--- class: 1
        # |   |   |--- feature_2 >  5316.50
        # |   |   |   |--- class: 0
        # |   |--- feature_2 >  7073.50
        # |   |   |--- feature_0 <= 60.50
        # |   |   |   |--- class: 1
        # |   |   |--- feature_0 >  60.50
        # |   |   |   |--- class: 1

    def forward(self, x):
        return self.feed_forward(x)

    def compute_TED(self, X):
        """
        compute the edit distance between the network's distilled decision tree and the original tree
        first distill a decision tree from the network
        then compute the edit distance between the distilled tree and the original tree using zss
        """
        # Helper function to convert a sklearn tree to a zss tree
        def sklearn_to_zss(node_id, tree):
            """Convert a sklearn decision tree node to a ZSS tree node with feature-only labels."""
            feature_index = tree.feature[node_id]
            node = Node(str(feature_index))  # Only the feature index as the label

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            if left_child != -1:
                node.addkid(sklearn_to_zss(left_child, tree))
            if right_child != -1:
                node.addkid(sklearn_to_zss(right_child, tree))

            return node

        def text_to_zss(text):
            """Parse a list of strings in format 'num_children:feature|threshold' into a zss.Node tree."""

            def parse_node(lines, idx):
                line = lines[idx]
                num_children_str, label = line.split(":", 1)
                num_children = int(num_children_str)
                node = Node(label)
                idx += 1
                for _ in range(num_children):
                    child, idx = parse_node(lines, idx)
                    node.addkid(child)
                return node, idx

            lines = text.strip().split("\n")
            return parse_node(lines, 0)[0]

        original_tree=text_to_zss(self.original_tree)
        self.freeze_model()
        self.eval()
        y_tree = self(X)
        y_tree = torch.where(y_tree > 0.5, 1, 0).detach().cpu().numpy()
        self.unfreeze_model()
        self.train()

        X_tree = X.cpu().detach().numpy()

        edit_distances = []

        for random_state in self.random_seeds:
            tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, random_state=random_state)
            tree.fit(X_tree, y_tree)
            zss_tree = sklearn_to_zss(tree.tree_.children_left[0], tree.tree_)
            # original_zss_tree = sklearn_to_zss(original_tree.tree_.children_left[0], original_tree.tree_)
            edit_distance = simple_distance(zss_tree, original_tree)
            edit_distances.append(edit_distance)
            del tree

        return np.mean(edit_distances)


    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """

        for param in self.feed_forward.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.feed_forward.parameters():
            param.requires_grad = True

    def freeze_bias(self):
        """
        Disable model updates by gradient-descent by freezing the biases.
        """
        for name, param in self.feed_forward.named_parameters():
            if 'bias' in name:
                param.requires_grad = False

    def reset_outer_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        """
        self.feed_forward.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def reset_surrogate_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        """
        self.surrogate_network.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def parameters_to_vector(self) -> torch.Tensor:
        """
        Convert model parameters to vector.
        """

        return parameters_to_vector(self.feed_forward.parameters())

    def vector_to_parameters(self, parameter_vector):
        """
        Overwrite the model parameters with given parameter vector.
        """
        vector_to_parameters(parameter_vector, self.feed_forward.parameters())
