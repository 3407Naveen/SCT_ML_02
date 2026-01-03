import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# Generate sample data mimicking Mall Customer Dataset
np.random.seed(42)
n_customers = 200

customer_ids = range(1, n_customers + 1)
genders = np.random.choice(['Male', 'Female'], n_customers)
ages = np.random.randint(18, 70, n_customers)
annual_incomes = np.random.randint(15, 137, n_customers)  # in k$
spending_scores = np.random.randint(1, 100, n_customers)

data = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income (k$)': annual_incomes,
    'Spending Score (1-100)': spending_scores
})

print("Sample Data:")
print(data.head())

# Preprocessing
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male: 1, Female: 0

# Select features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow method to find optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')
plt.close()

# Assuming optimal k is 5 based on elbow method (common for this dataset)
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to data
data['Cluster'] = y_kmeans

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

print("\nSilhouette Score:", silhouette_score(X, y_kmeans))

# Visualize clusters (using Age and Annual Income for 2D plot)
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(X[y_kmeans == i]['Age'], X[y_kmeans == i]['Annual Income (k$)'], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.savefig('customer_clusters.png')
plt.close()

# Summary of clusters
print("\nCluster Summary:")
for i in range(k):
    cluster_data = data[data['Cluster'] == i]
    print(f"Cluster {i}: {len(cluster_data)} customers")
    print(f"  Average Age: {cluster_data['Age'].mean():.2f}")
    print(f"  Average Annual Income: {cluster_data['Annual Income (k$)'].mean():.2f}")
    print(f"  Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.2f}")
    print()
