import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from preprocessing import preprocess_data

def rf(random_state, max_depth, max_leaf_nodes, n_estimators):
    print('Loading and preprocessing data...')
    data = pd.read_csv('rainfall_data.csv')
    processed_data = preprocess_data(data)

    target = processed_data['floodRisk'].astype(np.float32)
    features = processed_data.drop(['dateOfLoss', 'floodRisk'], axis=1)

    print('Scaling features...')
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features.astype(np.float32))
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_features['floodRisk'] = target

    print('Splitting data into train and test sets...')
    # print(scaled_features.head())

    Y = scaled_features['floodRisk'].values
    X = scaled_features.drop(['floodRisk'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    #model = Sequential(RandomForestClassifier(random_state=random_state))
    model = RandomForestClassifier(random_state=random_state)

    print('Training model...')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print('Evaluating model...')
    accuracy = accuracy_score(y_test, prediction)

    print('Classification...')
    print(classification_report(y_test, prediction))

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
            plt.savefig('confusion_matrix1.png')
            plt.close()

            return cm_normalized
        except Exception as e:
            print(f"Error in making confusion matrix: {str(e)}")

    print('Confusion Matrix...')
    print(confusion_matrix(y_test, prediction))
    plot_confusion_matrix(y_test, prediction)

    print('Accuracy...')
    print(accuracy)

rf(max_depth=10, n_estimators=150, random_state=30, max_leaf_nodes=None)

def rf_tune(random_state):
    print('Loading and preprocessing data...')
    data = pd.read_csv('rainfall_data.csv')
    processed_data = preprocess_data(data)

    target = processed_data['floodRisk'].astype(np.float32)
    features = processed_data.drop(['dateOfLoss', 'floodRisk'], axis=1)

    print('Scaling features...')
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features.astype(np.float32))
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_features['floodRisk'] = target

    print('Splitting data into train and test sets...')
    # print(scaled_features.head())

    Y = scaled_features['floodRisk'].values
    X = scaled_features.drop(['floodRisk'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    #model = Sequential(RandomForestClassifier(random_state=random_state))
    model = RandomForestClassifier(random_state=random_state)

    # params =  { 
    #     'n_estimators': [5, 10, 15, 20], 
    #     'max_features': ['sqrt', 'log2', None], 
    #     'max_depth': [1, 2, 3, 4],
    #     'criterion' :['gini', 'entropy']
    # }

    # Best Params...
    # {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 20}
    # Best estimator...
    # RandomForestClassifier(max_depth=4, n_estimators=20, random_state=10)

    # params = {
    #     'max_depth': [4, 5, 10, 20, 40], 
    #     'n_estimators': [20, 30, 40, 50]
    # }

    # Best Params...
    # {'max_depth': 10, 'n_estimators': 30}
    # Best estimator...
    # RandomForestClassifier(max_depth=10, n_estimators=30, random_state=10)
    
    params = {
        'max_depth': [8, 9, 10, 11, 12], 
        'n_estimators': [25, 30, 35]
    }

    # Best Params...
    # {'max_depth': 10, 'n_estimators': 30}
    # Best estimator...
    # RandomForestClassifier(max_depth=10, n_estimators=30, random_state=10)

    print('Running GridSearchCV...')
    grid = GridSearchCV(model, params, verbose=3)
    grid.fit(X_train, y_train)

    print('Best Params...')
    print(grid.best_params_)

    print('Best estimator...')
    print(grid.best_estimator_)

    # print('Training model...')
    # model.fit(X_train, y_train)
    # prediction = model.predict(X_test)

    # print('Evaluating model...')
    # accuracy = accuracy_score(y_test, prediction)

    # print('Classification...')
    # print(classification_report(y_test, prediction))

    # print('Accuracy...')
    # print(accuracy)

# rf_tune(random_state=10)