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
# Critic Network (surrogate estimating APL)
# ------------------------
class CriticAPL(nn.Module):
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
# True APL computation via scikit-learn
# ------------------------
def compute_true_apl(model, dataloader):
    model.eval()
    X, Y = [], []
    for x_batch, y_batch in dataloader:
        with torch.no_grad():
            y_pred = model(x_batch).round().squeeze().cpu().numpy()
        X.append(x_batch.cpu().numpy())
        Y.append(y_pred)
    X = np.vstack(X)
    Y = np.concatenate(Y)

    path_lengths = []

    for random_state in np.random.randint(1, 100, 10):
        tree = DecisionTreeClassifier(min_samples_leaf=50, random_state=random_state)
        tree.fit(X, Y)
        average_path_length = np.mean(np.sum(tree.tree_.decision_path(X), axis=1))
        path_lengths.append(average_path_length)

        del tree

    return np.mean(path_lengths)

# ------------------------
# Training Loop
# ------------------------
def cooling_schedule(epoch, lambda_init, lambda_target, total_epochs, alpha=10):
    """ Sigmoid annealing of lambda_apl """
    t = epoch / total_epochs
    return lambda_target + (lambda_init - lambda_target) * (1 / (1 + np.exp(alpha * (t - 0.5))))


def augment_data_with_dirichlet(X_train, theta_buffer, apl_buffer, model_template, device, num_new_samples=5):
    """Dirichlet-weighted interpolation of flattened model parameters + recomputed APL."""
    # Ensure all tensors in theta_buffer are on the correct device
    theta_buffer = [t.to(device) for t in theta_buffer]

    parameters = torch.stack(theta_buffer).to(device)  # shape [N, D]
    apls = torch.tensor(apl_buffer, dtype=torch.float32).to(device)

    alpha = [1.0] * len(parameters)
    weights = np.random.dirichlet(alpha, size=num_new_samples)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)  # shape [num_new_samples, N]

    synthetic_params = weights @ parameters  # shape [num_new_samples, D]
    synthetic_apls = []

    for flat_param in synthetic_params:
        new_model = unflatten_params(flat_param, lambda: model_template(input_dim=X_train.dataset[0][0].shape[0],
                                                                        hidden_dim=20, output_dim=1)).to(device)
        apl = compute_true_apl(new_model, X_train)
        synthetic_apls.append(apl)
        del new_model

    return list(synthetic_params), synthetic_apls



import matplotlib.pyplot as plt

def train(actor, critic, train_loader, dir, task_criterion,
          lambda_init=0.1, lambda_target=1.0,
          apl_update_every=25, critic_lr=1e-3, actor_lr=1e-3, epochs=10):

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    theta_buffer = []
    apl_buffer = []

    # Logging lists
    actor_losses = []
    critic_losses = []
    true_apls = []
    pred_apls = []
    accuracies = []

    actor_task_losses = []
    actor_apl_losses = []

    for epoch in range(epochs):
        lambda_apl = cooling_schedule(epoch, lambda_init, lambda_target, epochs)

        for step, (x, y) in enumerate(train_loader):
            actor.train()
            y_pred = actor(x).squeeze()
            task_loss = task_criterion(y_pred, y.float())

            theta_flat = flatten_params(actor)
            critic_apl_est = critic(theta_flat).squeeze()
            loss = task_loss + lambda_apl * critic_apl_est

            actor_task_losses.append(task_loss.item())
            actor_apl_losses.append((lambda_apl * critic_apl_est).item())

            # Backprop actor
            actor_opt.zero_grad()

            critic_apl_est.backward(retain_graph=True)
            apl_grad_norm = sum(p.grad.norm().item() for p in actor.parameters() if p.grad is not None)
            # print(f"[Step {step}] APL Grad Norm: {apl_grad_norm:.4e}")
            actor.zero_grad()  # before backwarding full loss

            loss.backward()
            actor_opt.step()

            # Store data for critic
            if step % apl_update_every == 0:
                true_apl = compute_true_apl(actor, train_loader)
                theta_vec = detach_flatten_params(actor).numpy()
                theta_tensor = torch.tensor(theta_vec, dtype=torch.float32)
                theta_buffer.append(theta_tensor)
                apl_buffer.append(true_apl)

                # Augment buffer with true Dirichlet samples
                aug_theta, aug_apl = augment_data_with_dirichlet(train_loader, theta_buffer, apl_buffer, ActorMLP,
                                                                 device, num_new_samples=10)
                # Ensure all tensors are on the same device
                theta_buffer = [t.to(device) for t in theta_buffer]
                aug_theta = [t.to(device) for t in aug_theta]

                # Stack tensors
                all_theta = torch.stack(theta_buffer + aug_theta).to(device)
                all_theta = torch.stack(theta_buffer + aug_theta).to(device)
                all_apl = torch.tensor(apl_buffer + aug_apl, dtype=torch.float32).unsqueeze(1).to(device)

                # Train critic more
                critic.train()
                for _ in range(3):  # train critic 3x per actor update
                    pred_apl = critic(all_theta)
                    critic_loss = nn.MSELoss()(pred_apl, all_apl)
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                if len(theta_buffer) > 20:
                    theta_buffer = theta_buffer[-10:]
                    apl_buffer = apl_buffer[-10:]

        # Evaluation at epoch end
        actor.eval()
        with torch.no_grad():
            true_apl = compute_true_apl(actor, train_loader)
            pred_apl = critic(flatten_params(actor).to(device)).item()

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
            true_apls.append(true_apl)
            pred_apls.append(pred_apl)
            accuracies.append(accuracy)

            print(f"[Epoch {epoch}] Task Loss: {task_loss.item():.4f}, "
                  f"APL Reg Loss: {(lambda_apl * critic_apl_est).item():.4f}, "
                  f"Lambda: {lambda_apl:.3f}, True APL: {true_apl:.2f}, "
                  f"Pred APL: {pred_apl:.2f}, Accuracy: {accuracy:.4f}")

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
    plt.plot(epochs_range, true_apls, label="True APL", color='orange')
    plt.plot(epochs_range, pred_apls, label="Predicted APL", color='green', linestyle='--')
    plt.title("True vs Predicted APL")
    plt.xlabel("Epoch")
    plt.ylabel("Average Path Length")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(dir,"apl_curve.png"))
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
    plt.plot(epochs_range, critic_losses, label="Critic (APL) Loss", color='red')
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
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)
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
    results_dir = os.path.join('ACresults', timestamp)
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
    critic = CriticAPL(param_dim).to(device)

    # ---------------------
    # Train with APL Regularization
    # ---------------------
    criterion = BCEWithLogitsLoss()
    train(actor,
          critic,
          train_loader,
          dir=results_dir,
          task_criterion=criterion,
          lambda_init=0.01,
          lambda_target=0.1,
          apl_update_every=25,
          critic_lr=1e-3,
          actor_lr=1e-3,
          epochs=30)
