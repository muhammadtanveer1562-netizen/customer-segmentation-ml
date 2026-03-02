# Customer Segmentation Dashboard
Unsupervised Machine Learning Project | K-Means Clustering | Streamlit Deployment

---

## 📌 Project Overview

This project applies K-Means clustering to segment customers based on their annual income and spending behavior.

The objective is to identify meaningful customer groups that can help businesses design targeted marketing strategies and improve decision-making.

The project also includes a fully interactive Streamlit dashboard for real-time customer segment prediction and visualization.

---

## 📊 Dataset

Dataset: Mall Customer Dataset (Kaggle)

Features Used:
- Annual Income (k$)
- Spending Score (1–100)

---

## 🧠 Machine Learning Approach

- Unsupervised Learning
- Feature Scaling using StandardScaler
- Optimal cluster selection using Elbow Method
- K-Means Clustering
- Cluster visualization (2D scatter plot)

---

## 📈 Business Insights

The model segments customers into five major categories:

- Premium Customers (High income, high spending)
- Budget Customers (Low income, low spending)
- Careful Customers (High income, low spending)
- Impulsive Customers (Low income, high spending)
- Average Customers

These segments allow businesses to create personalized marketing strategies.

---

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Streamlit

---

## 🚀 Application Features

- Interactive dashboard
- Cluster visualization
- Customer segment prediction
- Highlighted real-time customer positioning
- Business interpretation of clusters

---

## ▶️ How to Run the Project

1. Install dependencies:
   pip install requirements.txt

2. Run the application:
   python -m streamlit run app.py

---

## 📂 Project Structure

Task-2-Customer-Segmentation/
│
├── mall_customers.csv
├── task2.ipynb
├── kmeans_model.pkl
├── scaler.pkl
├── app.py

---

## 📌 Conclusion

This project demonstrates the power of unsupervised learning in uncovering hidden customer patterns and transforming raw data into strategic business intelligence.
