# 👥 Customer Segmentation with K-Means

## 🎯 Objective

Segment customers into distinct groups based on purchasing behavior and demographics, allowing the business to develop personalized marketing strategies and improve targeting.

---

## 📊 Dataset Overview

The dataset includes the following fields:
- `recency`: Days since last purchase
- `freq`: Number of purchases
- `item_quantity`: Quantity of items bought
- `total_spend`: Total spending
- `avg_spend`: Average spend per transaction
- `age`: Customer age

---

## 🧪 Methodology

### 1. Data Preprocessing
- Removed records with zero total_spend
- Converted phone numbers to string
- Detected and removed outliers using IQR method
- Scaled data using `StandardScaler` and `MinMaxScaler`

### 2. Clustering
- Applied **K-Means clustering**
- Used **Elbow Method** to determine optimal number of clusters (`Yellowbrick`)
- Labeled and analyzed customer groups

---

## 📈 Key Visualizations

- Boxplots of features to detect anomalies  
- Elbow curve to identify optimal clusters  
- Cluster-wise average profiles (e.g., total spend, frequency)

---

## 🔍 Cluster Insights (Example)
| Cluster | Recency | Frequency | Spend  | Suggested Strategy  |
|---------|---------|-----------|--------|---------------------|
| 0       | Low     | High      | High   | Loyalty program     |
| 1       | High    | Low       | Low    | Reactivation offer  |
| 2       | Medium  | Medium    | Medium | Maintain engagement |

---

## 💡 Business Value

- Identify high-value vs. at-risk customers
- Personalize communications by customer type
- Improve ROI of marketing campaigns

---

## 🧾 Files

- `Deepdive by K-mean.ipynb`: Full analysis and clustering pipeline
- `customer.csv`: Input dataset (not included here)