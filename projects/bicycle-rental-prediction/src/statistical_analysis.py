import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def statistical_analysis(df):
    os.makedirs('plots/statistical_analysis', exist_ok=True)

    print("Step 1: Identify significant features...")
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.savefig('plots/statistical_analysis/correlation_matrix_of_features.png')
    plt.close()

    print("Step 2: Understand correlations between features and the target variable...")
    correlations = df.corr()['cnt'].sort_values(ascending=False)
    print("Correlations with target variable (cnt):\n", correlations)

if __name__ == "__main__":
    df = pd.read_csv('data/trainDataset.csv')
    statistical_analysis(df)
