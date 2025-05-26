import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from zss import simple_distance, Node
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

original_tree_str = """2:2
2:1
0:-2
2:0
0:-2
0:-2
0:-2"""
# ccp_alpha = 0.01
# |--- feature_2 <= 5095.50
# |   |--- feature_1 <= 12.50
# |   |   |--- class: 0
# |   |--- feature_1 >  12.50
# |   |   |--- feature_0 <= 30.50
# |   |   |   |--- class: 0
# |   |   |--- feature_0 >  30.50
# |   |   |   |--- class: 1
# |--- feature_2 >  5095.50
# |   |--- class: 1

# ccp_alpha = 0.005
# 2:2
# 2:1
# 2:0
# 0:-2
# 2:3
# 0:-2
# 0:-2
# 2:0
# 0:-2
# 0:-2
# 0:-2

# |--- feature_2 <= 5095.50
# |   |--- feature_1 <= 12.50
# |   |   |--- feature_0 <= 33.50
# |   |   |   |--- class: 0
# |   |   |--- feature_0 >  33.50
# |   |   |   |--- feature_3 <= 1820.50
# |   |   |   |   |--- class: 0
# |   |   |   |--- feature_3 >  1820.50
# |   |   |   |   |--- class: 1
# |   |--- feature_1 >  12.50
# |   |   |--- feature_0 <= 30.50
# |   |   |   |--- class: 0
# |   |   |--- feature_0 >  30.50
# |   |   |   |--- class: 1
# |--- feature_2 >  5095.50
# |   |--- class: 1



# ------------------------
# Actor Network (your main model)
# ------------------------
class ActorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# Critic Network (surrogate estimating TED)
# ------------------------
class CriticTED(nn.Module):
    def __init__(self, param_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, theta_flat):
        return self.net(theta_flat)

# ------------------------
# Utility to flatten model parameters
# ------------------------
def flatten_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def detach_flatten_params(model):
    return torch.cat([p.detach().view(-1).cpu() for p in model.parameters()])

def unflatten_params(flat_params, model_template):
    """Creates a new model and loads parameters from flat vector"""
    new_model = model_template()
    offset = 0
    for p in new_model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset+numel].view_as(p))
        offset += numel
    return new_model

# ------------------------
# True TED computation via scikit-learn
# ------------------------
def sklearn_to_zss(node_id, tree):
    """Convert a sklearn decision tree node to a ZSS tree node using feature index only."""
    feature_index = tree.feature[node_id]
    node = Node(str(feature_index))

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != -1:
        node.addkid(sklearn_to_zss(left_child, tree))
    if right_child != -1:
        node.addkid(sklearn_to_zss(right_child, tree))

    return node

def text_to_zss(text):
    """Parse a text-encoded tree (DFS order) to zss.Node tree."""

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

def compute_true_TED(model, dataloader):
    """
    Compute Tree Edit Distance (TED) between model predictions and a fixed tree.
    """
    random_seeds = np.random.randint(1, 100, 10)
    original_tree= text_to_zss(original_tree_str)

    model.eval()
    was_training = model.training
    for p in model.parameters():
        p.requires_grad = False

    X_all, Y_all = [], []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            logits = model(x_batch).squeeze()
            preds = (logits > 0.5).float().cpu().numpy()
            X_all.append(x_batch.cpu().numpy())
            Y_all.append(preds)

    X_np = np.vstack(X_all)
    Y_np = np.concatenate(Y_all)

    edit_distances = []
    for seed in random_seeds:
        tree = DecisionTreeClassifier(min_samples_leaf=50, random_state=seed)
        tree.fit(X_np, Y_np)
        zss_tree = sklearn_to_zss(0, tree.tree_)
        edit_distance = simple_distance(zss_tree, original_tree)
        edit_distance = np.log1p(edit_distance)
        edit_distances.append(edit_distance)

    if was_training:
        model.train()
    for p in model.parameters():
        p.requires_grad = True

    return np.mean(edit_distances)

# ------------------------
# Training Loop
# ------------------------
def cooling_schedule(epoch, lambda_init, lambda_target, total_epochs, alpha=10):
    """ Sigmoid annealing of lambda_TED """
    t = epoch / total_epochs
    return lambda_target + (lambda_init - lambda_target) * (1 / (1 + np.exp(alpha * (t - 0.5))))


def augment_data_with_dirichlet(X_train, theta_buffer, ted_buffer, model_template, device, num_new_samples=5):
    """Dirichlet-weighted interpolation of flattened model parameters + recomputed TED."""
    # Ensure all tensors in theta_buffer are on the correct device
    theta_buffer = [t.to(device) for t in theta_buffer]

    parameters = torch.stack(theta_buffer).to(device)  # shape [N, D]
    # TEDs = torch.tensor(TED_buffer, dtype=torch.float32).to(device)

    alpha = [1.0] * len(parameters)
    weights = np.random.dirichlet(alpha, size=num_new_samples)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)  # shape [num_new_samples, N]

    synthetic_params = weights @ parameters  # shape [num_new_samples, D]
    synthetic_TEDs = []

    for flat_param in synthetic_params:
        new_model = unflatten_params(flat_param, lambda: model_template(input_dim=X_train.dataset[0][0].shape[0],
                                                                        hidden_dim=20, output_dim=1)).to(device)
        ted = compute_true_TED(new_model, X_train)
        synthetic_TEDs.append(ted)
        del new_model

    return list(synthetic_params), synthetic_TEDs


def post_pruning(X, y):
    """Minimal-complexity post-pruning for large decision trees. Given data set (X,y), train a decision tree classifier
    and compute the ccp_alphas from possible pruning paths. Do cross-validation with 5 folds and use one-standard-error
    rule to get the most parsimonous tree.

        Parameters
        ----------
        X: Input features

        y: Labels

        Returns
        -------
        ccp_alpha: Selected best alpha a*
    """
    # https://medium.com/swlh/post-pruning-decision-trees-using-python-b5d4bcda8e23
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = ccp_alphas[:-1]
    scores = []
    if len(ccp_alphas) != 0:
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
            score = cross_val_score(clf, X, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
            scores.append(score)

        # average over folds, fix sign of mse
        fold_mse = -np.mean(scores, 1)
        # select the most parsimonous model (highest ccp_alpha) that has an error within one standard deviation of
        # the minimum mse.
        # I.e. the "one-standard-error" rule (see ESL or a lot of other tibshirani / hastie notes on regularization)
        selected_alpha = np.max(ccp_alphas[fold_mse <= np.min(fold_mse) + np.std(fold_mse)])

        return selected_alpha

    else:
        return 0.0


def train(actor, critic, train_loader, dir, task_criterion,
          lambda_init=0.1, lambda_target=1.0,
          ted_update_every=25, critic_lr=1e-3, actor_lr=1e-3, epochs=10):

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    theta_buffer = []
    ted_buffer = []

    # Logging lists
    actor_losses = []
    critic_losses = []
    true_teds = []
    pred_teds = []
    accuracies = []

    actor_task_losses = []
    actor_ted_losses = []

    for epoch in range(epochs):
        lambda_TED = cooling_schedule(epoch, lambda_init, lambda_target, epochs)

        for step, (x, y) in enumerate(train_loader):
            actor.train()
            y_pred = actor(x).squeeze()
            task_loss = task_criterion(y_pred, y.float())

            theta_flat = flatten_params(actor)
            critic_TED_est = critic(theta_flat).squeeze()
            loss = task_loss + lambda_TED * critic_TED_est

            actor_task_losses.append(task_loss.item())
            actor_ted_losses.append((lambda_TED * critic_TED_est).item())

            # Backprop actor
            actor_opt.zero_grad()

            critic_TED_est.backward(retain_graph=True)
            TED_grad_norm = sum(p.grad.norm().item() for p in actor.parameters() if p.grad is not None)
            # print(f"[Step {step}] TED Grad Norm: {TED_grad_norm:.4e}")
            actor.zero_grad()  # before backwarding full loss

            loss.backward()
            actor_opt.step()

            # Store data for critic
            if step % ted_update_every == 0:
                true_ted = compute_true_TED(actor, train_loader)
                theta_vec = detach_flatten_params(actor).numpy()
                theta_tensor = torch.tensor(theta_vec, dtype=torch.float32)
                theta_buffer.append(theta_tensor)
                ted_buffer.append(true_ted)

                # Augment buffer with true Dirichlet samples
                aug_theta, aug_TED = augment_data_with_dirichlet(train_loader, theta_buffer, ted_buffer, ActorMLP,
                                                                 device, num_new_samples=10)
                # Ensure all tensors are on the same device
                theta_buffer = [t.to(device) for t in theta_buffer]
                aug_theta = [t.to(device) for t in aug_theta]

                # Stack tensors
                all_theta = torch.stack(theta_buffer + aug_theta).to(device)
                all_theta = torch.stack(theta_buffer + aug_theta).to(device)
                all_ted = torch.tensor(ted_buffer + aug_TED, dtype=torch.float32).unsqueeze(1).to(device)

                # Train critic more
                critic.train()
                for _ in range(3):  # train critic 3x per actor update
                    pred_TED = critic(all_theta)
                    critic_loss = nn.MSELoss()(pred_TED, all_ted)

                    # true_log_ted = torch.log1p(all_ted)
                    # pred_log_ted = torch.log1p(pred_TED)
                    # critic_loss = nn.MSELoss()(pred_log_ted, true_log_ted)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                if len(theta_buffer) > 20:
                    theta_buffer = theta_buffer[-10:]
                    ted_buffer = ted_buffer[-10:]

        # Evaluation at epoch end
        actor.eval()
        with torch.no_grad():
            true_ted = compute_true_TED(actor, train_loader)
            pred_TED = critic(flatten_params(actor).to(device)).item()

            # Accuracy
            correct = total = 0
            for x_batch, y_batch in train_loader:
                logits = actor(x_batch).squeeze()
                preds = (logits > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
            accuracy = correct / total

            # Logging
            actor_losses.append(task_loss.item())
            critic_losses.append(critic_loss.item() if 'critic_loss' in locals() else 0.0)
            true_teds.append(true_ted)
            pred_teds.append(pred_TED)
            accuracies.append(accuracy)

            print(f"[Epoch {epoch}] Task Loss: {task_loss.item():.4f}, "
                  f"TED Reg Loss: {(lambda_TED * critic_TED_est).item():.4f}, "
                  f"Lambda: {lambda_TED:.3f}, True TED: {true_ted:.2f}, "
                  f"Pred TED: {pred_TED:.2f}, Accuracy: {accuracy:.4f}")

    # ------------------------
    # Visualization after training
    # ------------------------
    epochs_range = range(epochs)

    plt.figure()
    plt.plot(epochs_range, accuracies, label="Accuracy", color='blue')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(dir,"accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, true_teds, label="True TED", color='orange')
    plt.plot(epochs_range, pred_teds, label="Predicted TED", color='green', linestyle='--')
    plt.title("True vs Predicted TED")
    plt.xlabel("Epoch")
    plt.ylabel("Tree Edit Distance (TED)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(dir,"TED_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, actor_losses, label="Actor (Task) Loss", color='purple')
    plt.title("Actor Task Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(dir,"actor_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, critic_losses, label="Critic (TED) Loss", color='red')
    plt.title("Critic Surrogate Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.savefig(os.path.join(dir,"critic_loss_curve.png"))
    plt.close()

    # Final distillation: Extract tree from final model
    actor.eval()
    X_np, Y_np = [], []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            logits = actor(x_batch).squeeze()
            preds = (logits > 0.5).float()
            X_np.append(x_batch.cpu().numpy())
            Y_np.append(preds.cpu().numpy())

    X_np = np.vstack(X_np)
    Y_np = np.concatenate(Y_np)

    # Train decision tree on neural network's predictions
    ccp_alpha = post_pruning(X_np, Y_np)
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_np, Y_np)

    from sklearn.tree import plot_tree

    plt.figure(figsize=(16, 8))
    plot_tree(tree, feature_names=[f"x{i}" for i in range(X_np.shape[1])],
              class_names=["0", "1"], filled=True, rounded=True)
    plt.title("Distilled Decision Tree from Neural Network")
    plt.savefig(os.path.join(dir,"distilled_tree.png"))
    plt.close()



if __name__ == '__main__':
    device = 'mps'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Create a new folder under ACresults
    results_dir = os.path.join('ACresults_TED', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results will be saved in: {results_dir}")
    # ---------------------
    # Load dataset
    # ---------------------
    dataset_dir = os.path.join('dataset', 'adult_income')
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(dataset_dir, 'val_data.csv'))
    test_df = pd.read_csv(os.path.join(dataset_dir, 'test_data.csv'))

    X_train = train_df.drop(columns=['income'])
    y_train = train_df['income']
    X_val = val_df.drop(columns=['income'])
    y_val = val_df['income']
    X_test = test_df.drop(columns=['income'])
    y_test = test_df['income']

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    # ---------------------
    # Data Loaders
    # ---------------------
    def make_loader(X, y, batch_size):
        tensor_X = torch.tensor(X, dtype=torch.float32).to(device)
        tensor_y = torch.tensor(y.values, dtype=torch.float32).to(device)
        dataset = TensorDataset(tensor_X, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


    train_loader = make_loader(X_train, y_train, 128)
    val_loader = make_loader(X_val, y_val, 128)
    test_loader = make_loader(X_test, y_test, 128)

    # ---------------------
    # Initialize models
    # ---------------------
    input_dim = X_train.shape[1]
    hidden_dim = 20
    actor = ActorMLP(input_dim, hidden_dim, 1).to(device)

    param_dim = sum(p.numel() for p in actor.parameters())
    critic = CriticTED(param_dim).to(device)

    # ---------------------
    # Train with TED Regularization
    # ---------------------
    criterion = BCEWithLogitsLoss()
    train(actor,
          critic,
          train_loader,
          dir=results_dir,
          task_criterion=criterion,
          lambda_init=0.01,
          lambda_target=0.1,
          ted_update_every=25,
          critic_lr=1e-3,
          actor_lr=1e-3,
          epochs=30)
