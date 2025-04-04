import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize(df):
    os.makedirs('plots/visualization', exist_ok=True)

    print("Step 1: Explore temporal trends (rentals by hour, day, month)...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='hr', y='cnt', data=df)
    plt.title('Bicycle Rentals by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Rentals')
    plt.savefig('plots/visualization/bicycle_rentals_by_hour.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='mnth', y='cnt', data=df)
    plt.title('Bicycle Rentals by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Rentals')
    plt.savefig('plots/visualization/bicycle_rentals_by_month.png')
    plt.close()

    print("Step 2: Examine the impact of weather conditions on rentals...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weathersit', y='cnt', data=df)
    plt.title('Bicycle Rentals by Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Number of Rentals')
    plt.savefig('plots/visualization/bicycle_rentals_by_weather_situation.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('data/trainDataset.csv')
    visualize(df)
