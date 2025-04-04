import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def eda(df):
    os.makedirs('plots/eda', exist_ok=True)
    
    print("Step 1: Understanding the distribution of the target variable...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cnt'], bins=30, kde=True)
    plt.title('Distribution of Bicycle Rentals')
    plt.xlabel('Number of Rentals')
    plt.ylabel('Frequency')
    plt.savefig('plots/eda/distribution_of_bicycle_rentals.png')
    plt.close()

    print("Step 2: Analyzing relationships between features and the target...")
    sns.pairplot(df, vars=['temp', 'atemp', 'hum', 'windspeed'], hue='cnt')
    plt.title('Relationships between Features and Target')
    plt.savefig('plots/eda/relationships_between_features_and_target.png')
    plt.close()

    print("Step 3: Identifying patterns and anomalies in the data...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['temp', 'atemp', 'hum', 'windspeed']])
    plt.title('Boxplots of Features to Identify Anomalies')
    plt.savefig('plots/eda/boxplots_of_features_to_identify_anomalies.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('data/trainDataset.csv')
    eda(df)
