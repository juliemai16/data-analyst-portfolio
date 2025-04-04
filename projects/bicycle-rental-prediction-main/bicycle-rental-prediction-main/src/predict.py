import pandas as pd
import joblib
from data_preprocessing import preprocess_data
from tensorflow.keras.models import load_model

def predict():
    print("Step 1: Loading new data for prediction...")
    new_data = pd.read_csv('data/submissionDataset.csv')  # Replace with actual new data file

    print("Step 2: Preprocessing new data...")
    preprocessor = joblib.load('models/best_preprocessor.pkl')  # Load the best preprocessor
    X_new, _ = preprocess_data(new_data, fit_transform=False, preprocessor=preprocessor)

    print("Step 3: Loading the best model for prediction...")
    best_model = joblib.load('models/best_model.pkl')  # Load the best model

    print("Step 4: Making predictions...")
    predictions = best_model.predict(X_new)

    # Step 5: Concatenating predictions with new data
    new_data['predictions'] = predictions

    # Step 6: Saving the updated new data file with predictions
    output_file = 'data/submissionDataset_with_predictions.csv'
    new_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    predict()
