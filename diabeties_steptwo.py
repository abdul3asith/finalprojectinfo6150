from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("diabeties_imputed.csv")


features = dataset[["Glucose", "BMI", "Age"]]

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(features)

dataset['Cluster'] = kmeans.labels_

# 5. Map the cluster labels to custom names
# cluster_names = {0: 'Non Diabeties', 1: 'Cluster B'}
#
# # 6. Assign names to the clusters
# dataset['Cluster Name'] = dataset['Cluster'].map(cluster_names)
#
# # 7. Visualize the clusters in 3D
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# # Scatter plot of the features with different colors for each cluster
# ax.scatter(dataset['Glucose'], dataset['BMI'], dataset['Age'], c=dataset['Cluster'], cmap='viridis', s=100)
#
# # Label the axes
# ax.set_xlabel('Glucose')
# ax.set_ylabel('BMI')
# ax.set_zlabel('Age')
#
# # Add a title
# ax.set_title('K-Means Clustering (3D visualization)')
#
# # Show the plot

# 4. Add cluster labels to the original DataFrame
# dataset['Cluster'] = kmeans.labels_

# 5. Calculate the average Glucose for each cluster
cluster_avg_glucose = dataset.groupby('Cluster')['Glucose'].mean()

# 6. Assign 'Diabetes' to the cluster with the higher average Glucose and 'No Diabetes' to the other
cluster_names = {0: 'No Diabetes', 1: 'Diabetes'} if cluster_avg_glucose[0] < cluster_avg_glucose[1] else {0: 'Diabetes', 1: 'No Diabetes'}

# 7. Assign names to the clusters
dataset['Cluster Name'] = dataset['Cluster'].map(cluster_names)

# 8. Add Outcome column (1 for 'Diabetes' and 0 for 'No Diabetes')
dataset['Outcome'] = dataset['Cluster'].map(lambda x: 1 if cluster_names[x] == 'Diabetes' else 0)

# Print the resulting DataFrame with the new 'Outcome' column

print(dataset)
dataset.to_csv("diabeties_cluster.csv", index=False)