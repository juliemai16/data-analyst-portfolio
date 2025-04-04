import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    print("Step 1: Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape {df.shape}")
    return df

def preprocess_data(df, fit_transform=True, preprocessor=None):
    print("Step 2: Preprocessing data...")

    print(" - Converting categorical features to string...")
    df['season'] = df['season'].astype(str)
    df['yr'] = df['yr'].astype(str)
    df['mnth'] = df['mnth'].astype(str)
    df['hr'] = df['hr'].astype(str)
    df['weekday'] = df['weekday'].astype(str)
    df['weathersit'] = df['weathersit'].astype(str)

    numerical_features = ['temp', 'atemp', 'hum']
    categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']

    print(" - Creating transformers for numerical and categorical features...")
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')  # Drop first to avoid multicollinearity

    if fit_transform:
        print(" - Applying fit_transform...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        processed_data = preprocessor.fit_transform(df)
    else:
        if preprocessor is None:
            raise ValueError("preprocessor cannot be None when fit_transform is False.")
        
        print(" - Applying transform...")
        processed_data = preprocessor.transform(df)

    print(f"Data preprocessing completed. Processed data shape: {processed_data.shape}")
    return processed_data.toarray(), preprocessor

# Example usage
if __name__ == "__main__":
    df = load_data('data/trainDataset.csv')
    processed_data, preprocessor = preprocess_data(df, fit_transform=True)
