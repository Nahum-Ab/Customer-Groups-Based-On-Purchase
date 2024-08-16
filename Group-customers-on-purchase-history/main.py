import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import joblib # For saving and loading the model

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f'Error: The file {file_path} was not found.')
        return None


def preprocess_data(data):
    # Selecting relevant features and handle missing values
    features = data[['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Gender']]
    features.dropna(inplace=True)

    # Convert categorical variable 'Gender' to numerical
    features['Gender'] = features['Gender'].map({'Male': 0, 'Female': 1})
    return features


# Function to scale the features
def scale_features(features):
    # Scaling the features using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

def determine_optimal_clusters(features_scaled):
    wcss = [] # with cluster sum of squares

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Function to create and train the K-means model
def perform_kmeans_clustering(features_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    return labels, kmeans


def evaluate_clustering(features_scaled, labels):
    # Evaluate the clustering using silhouette score
    score = silhouette_score(features_scaled, labels)
    print(f'Silhouette Score: {score: .2f}')

def visualize_clusters(data, features_scaled, clusters, kmeans):
    # Visualize the clustered data along with centroids.
    plt.figure(figsize=(10, 5))
    plt.scatter(features_scaled[:, 0],
                features_scaled[:, 1],
                c=clusters, cmap='viridis')

    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                s=300, c='red', label='Centroids')

    plt.title('Customer Segmentation')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.legend()
    plt.show()

    # Interactive visualization using Plotly
    df_plot = data.copy()
    df_plot['Cluster'] = clusters
    fig = px.scatter(df_plot, x='Annual Income (k$)', y='Spending Score (1-100)',
                     color='Cluster', hover_data=['Age', 'Gender'], title='Interactive Customer Segmentation')
    fig.show()


# Function to save the trained K-means model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f'Model saved as {filename}')
    return model

# The main function to execute the clustering process
def main():
    file_path = 'Mall_Customers.csv'
    data = load_data(file_path)

    if data is not None:
        features = preprocess_data(data)
        features_scaled = scale_features(features)
        determine_optimal_clusters(features_scaled)

        optimal_clusters = int(input("Enter the number of clusters you want to create: "))
        clusters, kmeans = perform_kmeans_clustering(features_scaled, optimal_clusters)

        data['Cluster'] = clusters
        evaluate_clustering(features_scaled, clusters)
        visualize_clusters(data, features_scaled, clusters, kmeans)

        save_choice = input("Do you want to save the model? (yes/no): ")
        if save_choice.lower() == 'yes':
            filename = input("Enter the filename to save the model: ")
            save_model(kmeans, filename)

if __name__ == "__main__":
    main()