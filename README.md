# PRODIGY_ML_02

# 🎯 Customer Segmentation with K-means Clustering

This project implements a K-means clustering algorithm to segment customers based on their purchasing behavior, analyzing features like annual income, spending score, age, and gender. By grouping customers into distinct clusters, businesses can better understand their customer base and tailor their marketing strategies accordingly.

## 🌟 Key Features

- **Data Preprocessing**: Handles missing values and converts categorical variables into numerical formats.
- **Feature Scaling**: Standardizes features to ensure they are on the same scale for accurate clustering.
- **Elbow Method**: Determines the optimal number of clusters using the Elbow method for visual identification of the best K value.
- **K-means Clustering**: Applies the K-means algorithm to segment customers into distinct clusters based on their similarities.
- **Evaluation**: Calculates the silhouette score to assess the quality of the clustering and ensure well-defined segments.
- **Visualization**: Provides both static and interactive visualizations using Matplotlib and Plotly to explore the customer segments.
- **Model Saving and Loading**: Allows saving and loading the trained K-means model for future use.

## 🛠️ Technologies Used

- **Python**: The primary programming language for implementing the K-means algorithm.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For creating static visualizations.
- **Scikit-learn**: For machine learning algorithms and model evaluation.
- **Plotly**: For interactive visualizations.
- **Joblib**: For saving and loading the trained K-means model.
