import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import os

# --------------------------- #
# 1. Load Preprocessed Dataset
# --------------------------- #

dataset_dir = os.path.join('dataset', 'adult_income')

train_df = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv'))
val_df = pd.read_csv(os.path.join(dataset_dir, 'val_data.csv'))
test_df = pd.read_csv(os.path.join(dataset_dir, 'test_data.csv'))

# Separate features and target
X_train = train_df.drop(columns=['income'])
y_train = train_df['income']

X_val = val_df.drop(columns=['income'])
y_val = val_df['income']

X_test = test_df.drop(columns=['income'])
y_test = test_df['income']


# ------------------------------- #
# 2. Train the Decision Tree Model
# ------------------------------- #

clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=50,  # Ensure enough samples per leaf
    random_state=42
)

clf.fit(X_train, y_train)

# ------------------------- #
# 3. Evaluate Model Performance
# ------------------------- #

def evaluate_model(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    print(f"\n Performance on {dataset_name}:")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred, pos_label=1):.4f}")
    print(f"Recall    : {recall_score(y, y_pred, pos_label=1):.4f}")
    print(f"F1-Score  : {f1_score(y, y_pred, pos_label=1):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

# Evaluate on Train, Validation, and Test sets
evaluate_model(clf, X_train, y_train, "Training Set")
evaluate_model(clf, X_val, y_val, "Validation Set")
evaluate_model(clf, X_test, y_test, "Test Set")

# ---------------------- #
# 4. Visualize the Tree (Optional)
# ---------------------- #

plt.figure(figsize=(40, 20))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['<=50K', '>50K'], fontsize=15)
plt.title("Decision Tree (Depth 5)")
plt.show()