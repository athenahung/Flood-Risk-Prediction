import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def preprocess_data(df):
    #df = pd.read_csv('rainfall_data.csv')
    # df.info()

    #Delete blank datapoints
    df = df.iloc[:360495]
    df = df.fillna(0)

    #Convert flood risk back to binary
    df['floodRisk'] = df['floodRisk'].astype(bool)

    #Handle missing data from baseFloodElevation and cause of Damage
    df['baseFloodElevation'] = df['baseFloodElevation'].interpolate()
    df['causeOfDamage'] = df['causeOfDamage'].interpolate()

    # df.info()

    #Exclude columns like data and floodRisk because these shouldn't get scaled/reduced, etc.
    exclude_cols = ['dateOfLoss', 'floodRisk']
    data_transform = df.drop(columns=exclude_cols)

    # print()
    # print("Data Before PCA:")
    # data_transform.info()

    #Perform dimensionality reduction
    pca = PCA(n_components=0.96)
    data_transform = pca.fit_transform(data_transform)
    data_transform = pd.DataFrame(data_transform, columns=[f'PC{i+1}' for i in range(data_transform.shape[1])])

    # print()
    # print("Data After PCA: ")
    # data_transform.info()

    #normalize data
    scaler = MinMaxScaler()
    data_transform = scaler.fit_transform(data_transform)
    data_transform = pd.DataFrame(data_transform, columns=[f'PC{i+1}' for i in range(data_transform.shape[1])])

    df = pd.concat([data_transform, df[exclude_cols].reset_index(drop=True)], axis=1)

    # print()
    # print("Final Dataframe after PCA and Normalization")
    # df.info()
    # print(df.head(15))
    return df

