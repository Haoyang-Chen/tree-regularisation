import copy
import torch
from dtreeviz.trees import *
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

np.random.seed(5555)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'mps'

import torch
from torch.utils.data import DataLoader, TensorDataset

def get_data_loader(X_train, y_train, X_test, y_test, X_val, y_val, X_type, y_type, batch_size):
    # Convert DataFrames to NumPy arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    X_val = X_val.values
    y_val = y_val.values

    # Convert NumPy arrays to tensors
    X_train = torch.tensor(X_train, dtype=X_type).to(device)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=y_type).to(device)
    X_test = torch.tensor(X_test, dtype=X_type).to(device)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=y_type).to(device)
    X_val = torch.tensor(X_val, dtype=X_type).to(device)
    y_val = torch.tensor(y_val.reshape(-1, 1), dtype=y_type).to(device)

    # Create DataLoaders
    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size)
    data_val = TensorDataset(X_val, y_val)
    data_val_loader = DataLoader(dataset=data_val, batch_size=batch_size)

    return data_train_loader, data_test_loader, data_val_loader


def dataloader_to_numpy(dataloader):
    """Convert data loader to numpy arrays.

        Parameters
        ----------
        dataloader: torch data loader


        Returns
        -------
        X : Features as numpy array

        y: Labels as numpy array
    """

    X = dataloader.dataset[:][0].detach().cpu().numpy()
    y = dataloader.dataset[:][1].detach().cpu().numpy()

    return X, y


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

def build_decision_tree(X_train, y_train, X_test, y_test,  min_samples_leaf=1):
    """Build tree given input data and save the corresponding tree plot and contour plot.

        Parameters
        ----------
        X_train: Training data features

        y_train: Labels for training data

        X_test: Test data features

        y_test: Labels for test data

        min_samples_leaf: Pre-pruning method, default 1 (no pruning)

        Returns
        -------
        accuracy: Accuracy measure of the decision tree using training and test set
    """

    ccp_alpha = post_pruning(X_train, y_train)
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)

    y_hat_tree = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat_tree)

    return accuracy


def plot_decision_tree(X_train, y_train):
    ccp_alpha = post_pruning(X_train, y_train)
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)

    # Convert X_train to DataFrame if it's a NumPy array
    feature_names = None
    if isinstance(X_train, np.ndarray):
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns

    # Plot the decision tree
    # Adjust figure size
    plt.figure(figsize=(16, 10))  # Increase figure size

    # Plot the decision tree with larger font sizes
    plot_tree(
        clf,
        filled=True,
        feature_names=feature_names,
        class_names=[str(c) for c in clf.classes_],
        fontsize=12,  # Increase font size
        proportion=True  # Normalize box sizes
    )

    plt.title("Decision Tree Visualization")
    plt.show()
    return clf




def augment_data_with_dirichlet(X_train, parameters, model, device, num_new_samples):
    """Draw new synthetic data for surrogate model using Dirichlet distribution.

        Parameters
        ----------
        X_train: Training data features

        parameters: Set of model parameters

        model: Target deep model

        device: Device, where the model is been trained (cpu or gpu)

        num_new_samples: Desired of new synthetic samples

        Returns
        -------
        parameters_new: New synthetic parameter set

        APLs_new: New synthetic APL estiamtes
    """

    parameters_new = []
    # APLs_new = []
    TEDs_new=[]

    alpha = [1] * len(parameters)
    samples = np.random.dirichlet(alpha, num_new_samples)
    parameters = torch.vstack(parameters).to(device)
    samples = torch.from_numpy(samples).float().to(device)
    parameters_ = samples @ parameters

    model.to(device)
    model.eval()

    for param in parameters_:
        model.vector_to_parameters(param)
        # APL = model.compute_APL(X_train)
        TED = model.compute_TED(X_train)

        parameters_new.append(param)
        # APLs_new.append(APL)
        TEDs_new.append(TED)

    del model
    del parameters_
    del samples

    # return parameters_new, APLs_new
    return parameters_new, TEDs_new


def augment_data_with_gaussian(X_train, model, device, size):
    """ DEPRECATED
    Draw new synthetic data for surrogate model using Gaussian distribution.

        Parameters
        ----------
        X_train: Training data features

        parameters: Set of model parameters

        model: Target deep model

        device: Device, where the model is been trained (cpu or gpu)

        num_new_samples: Desired of new synthetic samples

        Returns
        -------
        parameters_new: New synthetic parameter set

        APLs_new: New synthetic APL estiamtes
    """

    parameters = []
    APLs = []

    for _ in range(size):

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        for param in model_copy.feed_forward.parameters():
            param.data.requires_grad = False

            # variance: 0.1 - 0.3 times relative to the absolute value of the model parameter
            param_augmented = np.random.normal(param.data.cpu().numpy(), 0.1 * np.abs(param.data.cpu().numpy()))
            param.data = torch.tensor(param_augmented, dtype=torch.float).float().to(device)

        parameters.append(model_copy.get_parameter_vector)
        APLs.append(model_copy.compute_APL(X_train))

        del model_copy

    return parameters, APLs
