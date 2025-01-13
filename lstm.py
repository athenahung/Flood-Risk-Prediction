import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def lstm(time_step, max_depth, random_state):
    def prepare_sequences(data, time_step):
        """
        Prepare sequences for LSTM model with explicit data type handling
        """
        X, y = [], []
        data_array = data.astype(np.float32).values

        for i in range(len(data_array) - time_step):
            X.append(data_array[i:(i + time_step)])
            y.append(data_array[i + time_step][-1])  # floodRisk = last column

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y
    print("Loading and preprocessing data...")
    data = pd.read_csv('rainfall_data.csv')
    processed_data = preprocess_data(data)

    target = processed_data['floodRisk'].astype(np.float32)
    features = processed_data.drop(['dateOfLoss', 'floodRisk'], axis=1)

    print("Scaling features...")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features.astype(np.float32))
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_features['floodRisk'] = target

    print("Creating sequences...")
    # time_step = 10
    X, y = prepare_sequences(scaled_features, time_step)

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print("Building LSTM model...")
    n_features = X_train.shape[2]

    model = Sequential([
        Input(shape=(time_step, n_features)),
        LSTM(64, return_sequences=True, dtype=np.float32),
        LSTM(32, return_sequences=False, dtype=np.float32),
        Dense(16, activation='relu', dtype=np.float32),
        Dense(1, activation='sigmoid', dtype=np.float32)
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    print("Training the model...")
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64,
            verbose=1
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        raise

    def predict_flood_risk(new_data, model, scaler, time_step):
        """
        Predict flood risk for new data with explicit data type handling
        """
        try:
            # Preprocess
            processed_new_data = preprocess_data(new_data)
            new_features = processed_new_data.drop(['dateOfLoss', 'floodRisk'], axis=1)
            scaled_new_features = scaler.transform(new_features.astype(np.float32))

            # sequences
            X_new = []
            for i in range(len(scaled_new_features) - time_step):
                X_new.append(scaled_new_features[i:(i + time_step)])
            X_new = np.array(X_new, dtype=np.float32)

            # predict
            predictions = model.predict(X_new)
            processed_new_data.loc[time_step:, 'predicted_floodRisk'] = predictions

            return processed_new_data

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    try:
        new_data = pd.read_csv('new_rainfall_data.csv')
        predictions_df = predict_flood_risk(new_data, model, scaler)
        predictions_df.to_csv('predicted_flood_risk.csv', index=False)
        print("Predictions saved to 'predicted_flood_risk.csv'")
    except FileNotFoundError:
        print("No new data file found. Model is ready for predictions.")
    #metrics
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    ##############################################################################

    def get_binary_predictions(model, X_test, threshold=0.5):
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        return y_pred

    def plot_confusion_matrix(y_true, y_pred):
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                        xticklabels=['No Flood', 'Flood'],
                        yticklabels=['No Flood', 'Flood'])

            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            plt.close()

            return cm_normalized
        except Exception as e:
            print(f"Error in making confusion matrix: {str(e)}")

    def create_decision_tree_visualization(X_train, y_train, feature_names):
        try:
            # Feature matching
            n_features = X_train.shape[1]
            if len(feature_names) != n_features:
                print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features ({n_features})")
                feature_names = [f'feature_{i}' for i in range(n_features)]

            # Train dc
            dt_model = DecisionTreeClassifier(max_depth, random_state)
            dt_model.fit(X_train, y_train)

            # plt
            plt.figure(figsize=(20, 10))
            plot_tree(dt_model,
                    feature_names=feature_names,
                    class_names=['No Flood', 'Flood'],
                    filled=True,
                    rounded=True,
                    fontsize=10)

            plt.title('Decision Tree Visualization')
            plt.tight_layout()
            plt.savefig('decision_tree.png', bbox_inches='tight', dpi=300)
            plt.close()

            return dt_model

        except Exception as e:
            print(f"Error creating decision tree visualization: {str(e)}")
            return None
    print("\nGenerating visualizations...")

    y_pred = get_binary_predictions(model, X_test)
    y_test_binary = (y_test >= 0.5).astype(int)

    # confusion matrix
    cm = plot_confusion_matrix(y_test_binary, y_pred)
    print("Confusion matrix saved as 'confusion_matrix.png'")
    try:
        feature_names = scaled_features.columns[:-1].tolist()

        X_train_dt = X_train[:, 0, :]
        y_train_binary = (y_train >= 0.5).astype(int)

        print(f"Feature names: {feature_names}")
        print(f"X_train_dt shape: {X_train_dt.shape}")
        dt_model = create_decision_tree_visualization(X_train_dt, y_train_binary, feature_names)
        if dt_model is not None:
            print("Decision tree visualization saved as 'decision_tree.png'")

    except Exception as e:
        print(f"Error in decision tree prep: {str(e)}")
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred))

    # print()
    # print('------------FINAL RESULTS------------')
    # print(f'Test Loss: {test_loss}')
    # print(f'Test Accuracy: {test_accuracy}')
    # print(f'True Positives {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}')

    return test_loss, test_accuracy, cm
