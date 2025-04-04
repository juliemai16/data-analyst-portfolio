# Bicycle Rental Prediction Project

## Introduction

The goal of this project is to predict the number of bicycle rentals per hour based on various features such as weather conditions, time of day, and other relevant factors.

## Project Overview

In this project, we develop and evaluate multiple machine-learning models to accurately forecast bicycle rental demand. The models include:
- TensorFlow neural network
- LightGBM
- Random Forest
- Gradient Boosting
- XGBoost

## Project Structure
```
bicycle-rental-prediction/
│
├── data/
│   ├── trainDataset.csv                              # Training dataset
│   ├── submissionDataset.csv                         # Testing dataset
│   └── submissionDataset_with_predictions.csv        # New dataset for making predictions
│
├── plots/
│   ├── eda/                                          # Directory to save EDA plots
│   ├── visualization/                                # Directory to save data visualization plots
│   └── statistical_analysis/                         # Directory to save statistical analysis plots
│
├── src/
│   ├── __init__.py                                   # Init file to treat the directory as a package
│   ├── data_preprocessing.py                         # Script for data preprocessing
│   ├── eda.py                                        # Script for exploratory data analysis
│   ├── modeling.py                                   # Script for model definitions and hyperparameter tuning
│   ├── train_and_evaluate.py                         # Script to train and evaluate models, selecting the best one
│   ├── predict.py                                    # Script to make predictions using the best model
│   ├── visualization.py                              # Script for data visualization
│   └── statistical_analysis.py                       # Script for statistical analysis
│
├── requirements.txt                                  # File listing the required Python packages
├── README.md                                         # Project documentation
└── .gitignore                                        # File specifying which files to ignore in version control

```

## Motivation

Accurate prediction of bicycle rentals can help in resource allocation, demand forecasting, and improving the efficiency of bike-sharing systems.

## Setup

### Environment Setup

To set up your environment, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/juliemai16/bicycle-rental-prediction.git
    cd bicycle-rental-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Data

### Data Collection

The dataset contains hourly rental data from the bike-sharing system. The features include:
- `id`: Record ID
- `season`: Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)
- `yr`: Year (0: 2011, 1: 2012)
- `mnth`: Month (1 to 12)
- `hr`: Hour (0 to 23)
- `holiday`: Holiday (0: No, 1: Yes)
- `weekday`: Day of the week
- `workingday`: Working day (0: No, 1: Yes)
- `weathersit`: Weather situation (1 to 4)
- `temp`: Normalized temperature
- `atemp`: Normalized feeling temperature
- `hum`: Normalized humidity
- `cnt`: Count of total rental bikes

### Data Preprocessing

Data preprocessing includes handling missing values, encoding categorical variables, and scaling numerical features.

## Exploratory Data Analysis (EDA)

### EDA involves:
- Understanding the distribution of the target variable
- Analyzing relationships between features and the target
- Identifying patterns and anomalies in the data

EDA scripts are located in `src/eda.py` and save figures in `plots/eda`.

## Data Visualization

Data visualization techniques are used to:
- Visualize the distribution of bicycle rentals
- Explore temporal trends (e.g., rentals by hour, day, month)
- Examine the impact of weather conditions on rentals

Visualization scripts are located in `src/visualization.py` and save figures in `plots/visualization`.

## Statistical Analysis

Statistical analysis is performed to:
- Identify significant features
- Understand correlations between features and the target variable

Statistical analysis scripts are located in `src/statistical_analysis.py` and save figures in `plots/statistical_analysis`.

## Modeling

### Model Architecture

We implement and compare several machine learning models, including:
- A neural network using TensorFlow
- Gradient boosting models (LightGBM, XGBoost)
- Random Forest

### Model Training

Models are trained using the preprocessed data. Key steps include:
- Splitting data into training and validation sets
- Hyperparameter tuning
- Early stopping to prevent overfitting

### Evaluation Metrics

Model performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Deployment

### Deployment Instructions

To deploy the best model for making predictions on new data:

1. Ensure the preprocessed data is available.
2. Load the saved model and preprocessor.
3. Use the model to make predictions on the new data.


## Results

### Performance Evaluation

Model performance is summarized with metrics and visualizations. Key insights include:
- Comparison of model performance
- Analysis of prediction accuracy

### Discussion

Discuss the results, including:
- Model strengths and weaknesses
- Potential improvements
- Implications for real-world applications

## Contributing

### Contribution Guidelines

To contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [Kaggle Competition: Proton X TF 09 - Bài toán dự đoán xe đạp](https://www.kaggle.com/competitions/proton-x-tf-09-bai-toan-du-doan-xe-dap/overview)
- Relevant research papers and articles on bicycle rental prediction
