import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def preprocess_data(df):
    debug = False

    if debug:
        print('original data:')
    df = pd.read_csv('rainfall_data.csv')
    if debug:
        df.info()
        print()

    # Handle missing data from baseFloodElevation and cause of Damage
    if debug:
        print('interpolate')
    df = df.iloc[:360495]
    df['baseFloodElevation'] = df['baseFloodElevation'].interpolate()
    df['causeOfDamage'] = df['causeOfDamage'].interpolate()
    if debug:
        df.info()
        print()

    # Convert flood risk to float
    if debug:
        print('convert floodRisk to float' if debug else '', end='')
    df['floodRisk'] = df['floodRisk'].astype('float')
    if debug:
        df.info()
        print()

    # Exclude columns like data and floodRisk because these shouldn't get scaled/reduced, etc.
    if debug:
        print('exclude dataOfLoss, floodRisk')
    exclude_cols = ['dateOfLoss', 'floodRisk']
    data_transform = df.drop(columns=exclude_cols)

    # Normalize data
    scaler = MinMaxScaler()
    data_transform = scaler.fit_transform(data_transform)
    data_transform = pd.DataFrame(data_transform, columns=['baseFloodElevation', 'lowestFloorElevation', 'causeOfDamage', 'precipitation'])

    df = pd.concat([data_transform, df[exclude_cols].reset_index(drop=True)], axis=1)

    if debug:
        print('\nFinal Dataframe after Normalization')
        df.info()
        print('\nFirst 15 rows')
        print(df.head(15))

    return df

# Step 1: Load the preprocessed data (replace 'data.csv' with your dataset file)
def load_data():
    data = pd.read_csv('rainfall_data.csv')
    processed_data = preprocess_data(data)
    return processed_data

def split_data(data, feature_columns=['baseFloodElevation', 'lowestFloorElevation', 'causeOfDamage', 'precipitation'], target_column='floodRisk'):
    # Extract features and labels
    X = data[feature_columns]
    y = data[target_column].astype(int)  # Convert boolean to int (0 or 1)

    # Split into training and test sets
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 3: Train the SVM model
def train_svm(X_train, y_train, kernel='rbf', C=0.5, gamma='scale'):
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight='balanced')
    svm_model.fit(X_train, y_train)
    return svm_model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    return y_pred, conf_matrix

# Step 5: Visualize the confusion matrix
def plot_normalized_confusion_matrix(y_true, y_pred, title='Normalized Confusion Matrix'):
    """
    Plots a confusion matrix with normalized (decimal) values.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        title: Title for the confusion matrix plot.
    """
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# Main function
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    data = load_data()
    print(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the model
    print("Training the SVM model...")
    svm_model = train_svm(X_train, y_train, C=0.9)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred, conf_matrix = evaluate_model(svm_model, X_test, y_test)
    print('\n====================================================================================\n')

    # Visualize the confusion matrix
    print("Visualizing the confusion matrix...")
    plot_normalized_confusion_matrix(y_test, y_pred, title="Normalized Confusion Matrix")