import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter

from networks_EXAI_test import TreeNet
from utils_EXAI import *
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score


np.random.seed(5555)
torch.random.manual_seed(5255)

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label',
                        required=False,
                        type=str,
                        default='no_label',
                        help='Additional label as postfix to the directory path name to indicate this run')


    parser.add_argument('--lambda_init',
                        required=False,
                        type=float,
                        default=0.5,
                        help='Initial lambda value as regularisation term')

    parser.add_argument('--lambda_target',
                        required=False,
                        type=float,
                        default=0.5,
                        help='Target lambda value as regularisation term')

    parser.add_argument('--ep',
                        required=False,
                        default=20,
                        type=int,
                        help='Total number of epochs, default 1000 (300 warm up + 700 regularisation)')

    parser.add_argument('--min_samples_leaf',
                        required=False,
                        default=50,
                        type=int,
                        help='Minimum samples leaf for pre-pruning, default 5')

    parser.add_argument('--batch',
                        default=128,
                        required=False,
                        help='Batch size, default 1024')

    return parser


def resample_data():

    #resample 5000 samples from training set
    dataset_dir = os.path.join('dataset', 'adult_income')
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv'))
    X_train = train_df.drop(columns=['income'])
    y_train = train_df['income']

    # Create a random subset of 5000 samples
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=5000, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_train_subset, y_train_subset, test_size=0.20, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float).to(device)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float).to(device)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=64)

    return data_train_loader, data_test_loader


def train(data_train_loader, data_test_loader, data_val_loader, path):

    model = TreeNet(input_dim=dim, min_samples_leaf=args.min_samples_leaf)
    model.to(device)

    total_num_epochs = args.ep
    epochs_warm_up = 5
    lambda_init = args.lambda_init
    lambda_ = lambda_init

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.1238, dtype=torch.float64))
    optimizer = Adam(model.feed_forward.parameters(), lr=1e-3)

    TEDs_truth=[]

    training_loss_without_reg = []
    val_loss = []
    training_accuracy = []
    tree_accuracy = []

    lambdas = [lambda_]

    x_iter_warm_up = 0
    iters_per_epoch = 0


    for epoch in range(total_num_epochs):
        model.train()
        batch_loss_val = []
        batch_loss_without_reg = []
        batch_TED_loss = []
        batch_accuracy = []

        for (x, y) in data_train_loader:

            y_hat = model(x)

            if epoch > (epochs_warm_up - 1): # regularisation phase
                omega = model.compute_TED(data_train_loader.dataset[:][0])
                loss = 2*criterion(input=y_hat, target=y) + lambda_ * omega

            else: # warm-up phase
                loss = 2*criterion(input=y_hat, target=y)
                x_iter_warm_up += 1

            loss_without_reg = criterion(input=y_hat, target=y)  # Only for plotting, not for optimisation
            batch_loss_without_reg.append(float(loss_without_reg))
            del loss_without_reg

            iters_per_epoch += 1 if epoch == 0 else 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_train_loader_new, _ = resample_data()

            TED = model.compute_TED(data_train_loader_new.dataset[:][0])
            batch_TED_loss.append(TED*lambda_)
            TEDs_truth.append(TED)

            del x, y

        # if epoch > 0 and epoch % 10 == 0:  # snapshots of the resulting tree
        #     if not os.path.exists('models'):
        #         os.makedirs('models')
        #     torch.save(model.state_dict(), f'models/model_snapshot_{epoch}.pth')
        #     model.eval()
        #     model.train()

        # Validation
        model.eval()
        for (x, y) in data_val_loader:
            y_hat = model(x)
            loss = criterion(input=y_hat, target=y)
            batch_loss_val.append(float(loss))
            y_hat = torch.where(y_hat > 0.5, 1, 0).cpu().numpy()
            y = y.detach().cpu().numpy()
            batch_accuracy.append(accuracy_score(y, y_hat))

            del x, y

        print(f'Epoch: [{epoch + 1}/{total_num_epochs}, Loss: {np.array(batch_loss_without_reg).mean():.4f}, TED: {np.array(batch_TED_loss).mean():.4f}]')

        training_loss_without_reg.append(np.array(batch_loss_without_reg).mean())
        val_loss.append(np.array(batch_loss_val).mean())
        training_accuracy.append(np.array(batch_accuracy).mean())

        data_train_loader_new, data_test_loader_new = resample_data()
        X_train_new, y_train_new = dataloader_to_numpy(data_train_loader_new)
        X_test_new, y_test_new = dataloader_to_numpy(data_test_loader_new)

        ccp_alpha = post_pruning(X_train_new, y_train_new)
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, min_samples_leaf=args.min_samples_leaf)
        y_hat_ = model(data_train_loader_new.dataset[:][0])
        y_hat_ = torch.where(y_hat_ > 0.5, 1, 0).detach().cpu().numpy()
        dt.fit(X_train_new, y_hat_)
        acc = accuracy_score(y_test_new, dt.predict(X_test_new))
        tree_accuracy.append(acc)

    # PLOTS

    fig = plt.figure()
    plt.plot(range(0, len(training_loss_without_reg)), training_loss_without_reg, label='Training loss')
    plt.plot(range(0, len(val_loss)), val_loss, label='Validation loss')
    plt.xlabel(f'epochs ({iters_per_epoch} iterations per epoch)')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.title('Training loss')
    fig.tight_layout()
    fig.savefig(f'{path}/training_loss.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(TEDs_truth)), TEDs_truth, color='y', label='true TED')
    plt.xlabel('iterations')
    plt.ylabel('TED')
    plt.legend()
    plt.grid()
    plt.title(f'TED estimates')
    fig.tight_layout()
    fig.savefig(f'{path}/TED_estimates.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(tree_accuracy)), tree_accuracy, color='b', label='Accuracy Decision Trees')
    plt.plot(range(0, len(training_accuracy)), training_accuracy, color='r', label='Accuracy Network')
    plt.xlabel(f'epochs ({iters_per_epoch} iterations per epoch)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.title(f'Accuracy')
    fig.tight_layout()
    fig.savefig(f'{path}/accuracy.png')
    plt.close(fig)

    # for i, value in enumerate(surrogate_training_loss):
    #     writer.add_scalar(f'Surrogate Training/Loss of surrogate training after epoch {i}', value, i)
    #
    # for i, value in enumerate(training_loss_without_reg):
    #     writer.add_scalar(f'Training loss without regularisation', value, i)
    #
    # for i, value in enumerate(APL_predictions):
    #     writer.add_scalar(f'APL Predictions', value, i)
    #
    # for i, value in enumerate(surrogate_training_loss):
    #     writer.add_scalar(f'Surrogate Training Loss', value, i)
    #     writer.add_scalar(f'Surrogate Training Loss', value, i)


    return model, criterion

def evaluate_metrics(y_true, y_pred):
    """
    Calculate and return precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc, cm, cr

def print_metrics(accuracy, precision, recall, f1, roc_auc, cm, cr, dataset_name):
    """
    Prints the performance metrics in a structured format.
    """
    print(f"\n{'='*10} {dataset_name} Performance Metrics {'='*10}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("="*50)

def init(path, data_train_loader, data_test_loader, data_val_loader):
    global X_train
    global y_train
    global X_test
    global y_test

    ############# Training ######################
    print('Training'.center(len('Training') + 2).center(30, '='))
    model, criterion = train(data_train_loader, data_test_loader, data_val_loader, path)

    ############# Evaluation #####################
    print('Test'.center(len('Test') + 2).center(30, '='))
    model.eval()

    X_train_temp = []  # Collect training data for evaluation
    y_train_temp = []

    y_train_NN_predicted = []
    y_test_NN_predicted = []

    with torch.no_grad():
        # Test with training data
        for (x, y) in data_train_loader:
            X_train_temp.append(x)
            y_train_temp.append(y)

            y_hat = model(x)
            y_train_NN_predicted.append(y_hat)

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_NN_predicted = torch.cat(y_train_NN_predicted)
        y_train_NN_predicted = torch.where(y_train_NN_predicted > 0.5, 1, 0).detach().cpu().numpy()

        # Test with test data
        for (x, y) in data_test_loader:
            y_hat = model(x)
            y_test_NN_predicted.append(y_hat)

        y_test_NN_predicted = torch.cat(y_test_NN_predicted)
        y_test_NN_predicted = torch.where(y_test_NN_predicted > 0.5, 1, 0).detach().cpu().numpy()

        # Compute performance metrics
        accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, cm_train, cr_train = evaluate_metrics(y_train_temp, y_train_NN_predicted)

        accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, cm_test, cr_test = evaluate_metrics(y_test, y_test_NN_predicted)

        # Print metrics for Training and Test Data
        print_metrics(accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, cm_train, cr_train, "Training_NN")
        print_metrics(accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, cm_test, cr_test, "Test_NN")

    data_train_loader_new, data_test_loader_new = resample_data()
    X_train_new, y_train_new = dataloader_to_numpy(data_train_loader_new)
    y_train_predicted_ = model(data_train_loader_new.dataset[:][0])
    y_train_predicted_ = torch.where(y_train_predicted_ > 0.5, 1, 0).detach().cpu().numpy().reshape(-1)

    _ = plot_decision_tree(X_train_new, y_train_predicted_)

    del model



if __name__ == '__main__':

    # device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    device = 'mps'

    args = parser().parse_args()

    # dataset
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

    data_train_loader, data_test_loader, data_val_loader = get_data_loader(X_train, y_train, X_test, y_test, X_val,
                                                                           y_val, torch.float, torch.float, args.batch)
    # dataset end

    dim = X_train.shape[1]

    dir_name = f'tree_reg_train_{args.lambda_init}_{args.lambda_target}_{args.label}'

    fig_path = f'figures/{dir_name}'
    tb_logs_path = f'runs/{dir_name}'

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    init(fig_path, data_train_loader, data_test_loader, data_val_loader)
