# Customer Segmentation using Machine Learning

## 📌 Overview
This project applies **K-Means** and **Hierarchical Clustering** to segment customers based on **Annual Income** and **Spending Score**.

## 📊 Dataset
The dataset (`Mall_Customers.csv`) contains customer information such as:
- Customer ID
- Age
- Gender
- Annual Income
- Spending Score

## 🛠️ Technologies Used
- Python
- NumPy, Pandas
- Scikit-Learn
- Matplotlib, Seaborn
- Scipy (for hierarchical clustering)

## 📌 Steps Followed
1. **Preprocessing** – Data selection and scaling.
2. **K-Means Clustering** – Elbow Method for optimal K value.
3. **Hierarchical Clustering** – Dendrogram analysis.
4. **Visualization** – Clustered data plotted using Matplotlib/Seaborn.
5. **Evaluation** – Silhouette Score for model performance.

## 🚀 Running the Code
1. Clone this repository:
   git clone https://github.com/QuantumCoderrr/Customer-Segmentation-ML.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the script:
   python src/clustering.py

## 📌 Results
The dataset is clustered into 5 groups based on customer spending patterns.
The project successfully segments customers using K-Means and Hierarchical Clustering.

## 📸 Visualizations
