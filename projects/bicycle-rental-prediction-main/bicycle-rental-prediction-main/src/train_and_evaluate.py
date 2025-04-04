import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm
import joblib
from data_preprocessing import load_data, preprocess_data
from model import create_tf_model, create_lgb_model, create_rf_model, create_gb_model, create_xgb_model
from sklearn.metrics import mean_absolute_error

def train_and_evaluate():
    print("Starting training process...")

    df = load_data('data/trainDataset.csv')
    X = df.drop(columns=['cnt'])
    y = df['cnt']

    print("Step 8: Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    print("Step 9: Preprocessing training and validation data...")
    X_train, preprocessor = preprocess_data(X_train, fit_transform=True)
    X_val, _ = preprocess_data(X_val, fit_transform=False, preprocessor=preprocessor)
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'models/best_preprocessor.pkl')
    print("Preprocessor saved successfully.")

    print("Step 10: Training TensorFlow model...")
    tf_model = create_tf_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tf_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
    tf_model.save('models/tf_model.h5')
    print("TensorFlow model trained and saved successfully.")

    print("Step 11: Training LightGBM model...")
    lgb_model = create_lgb_model()
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lightgbm.early_stopping(stopping_rounds=10)])
    joblib.dump(lgb_model, 'models/lgb_model.pkl')
    print("LightGBM model trained and saved successfully.")

    print("Step 12: Training Random Forest model...")
    rf_model = create_rf_model()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("Random Forest model trained and saved successfully.")

    print("Step 13: Training Gradient Boosting model...")
    gb_model = create_gb_model()
    gb_model.fit(X_train, y_train)
    joblib.dump(gb_model, 'models/gb_model.pkl')
    print("Gradient Boosting model trained and saved successfully.")

    print("Step 14: Training XGBoost model...")
    xgb_model = create_xgb_model()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=10)
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    print("XGBoost model trained and saved successfully.")

    # Evaluating model performance on validation set and selecting the best model
    print("Step 15: Evaluating model performance on training set...")
    tf_val_predictions = tf_model.predict(X_val)
    lgb_val_predictions = lgb_model.predict(X_val)
    rf_val_predictions = rf_model.predict(X_val)
    gb_val_predictions = gb_model.predict(X_val)
    xgb_val_predictions = xgb_model.predict(X_val)

    tf_val_mae = mean_absolute_error(y_val, tf_val_predictions)
    lgb_val_mae = mean_absolute_error(y_val, lgb_val_predictions)
    rf_val_mae = mean_absolute_error(y_val, rf_val_predictions)
    gb_val_mae = mean_absolute_error(y_val, gb_val_predictions)
    xgb_val_mae = mean_absolute_error(y_val, xgb_val_predictions)

    print(f"TensorFlow Validation MAE: {tf_val_mae}")
    print(f"LightGBM Validation MAE: {lgb_val_mae}")
    print(f"Random Forest Validation MAE: {rf_val_mae}")
    print(f"Gradient Boosting Validation MAE: {gb_val_mae}")
    print(f"XGBoost Validation MAE: {xgb_val_mae}")

    # Selecting the best model based on Validation MAE
    best_model = None
    min_val_mae = float('inf')
    models = {
        "TensorFlow": (tf_model, tf_val_mae),
        "LightGBM": (lgb_model, lgb_val_mae),
        "Random Forest": (rf_model, rf_val_mae),
        "Gradient Boosting": (gb_model, gb_val_mae),
        "XGBoost": (xgb_model, xgb_val_mae)
    }

    for model_name, (model, val_mae) in models.items():
        if val_mae < min_val_mae:
            min_val_mae = val_mae
            best_model = model_name

    print(f"Best model selected: {best_model}")

    # Save the best model
    if best_model == "TensorFlow":
        joblib.dump(preprocessor, 'models/best_preprocessor.pkl')  # Save preprocessor for TensorFlow model
        print("Best preprocessor saved successfully.")
    best_model_obj, _ = models[best_model]
    joblib.dump(best_model_obj, 'models/best_model.pkl')
    print("Best model saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()
