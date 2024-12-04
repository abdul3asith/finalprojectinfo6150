import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from sklearn.naive_bayes import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# from sklean.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# from keras.models import Sequential
# from keras.layers import Dense, Input

dataset = pd.read_csv("Loss_of_Customers_Cleaned.csv")

X = dataset.loc[:,dataset.columns != 'loss_of_customer']
y = dataset["loss_of_customer"]
train_x,test_x,train_y,test_y = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=9)

norm_feat_df = (X - X.mean()) / X.std()
model = PCA()
model.fit(norm_feat_df)

print(model.get_covariance())

pc_feature_relationship = pd.DataFrame(model.components_, columns=norm_feat_df.columns)
print(pc_feature_relationship)

x_axis = range(model.n_components_)
plt.plot(x_axis, np.cumsum(model.explained_variance_ratio_), marker="o")
plt.xlabel('Principal component')
plt.ylabel('Cumulative sum of explained variance')
plt.xticks(x_axis)
plt.show()

knn.fit(X, y)
y_pred = knn.predict(X)

accuracy = accuracy_score(y, y_pred)
print('Accuracy by Original Features:', accuracy )

pca = PCA(n_components=3)
pca.fit(norm_feat_df)

transformed_features_df = pca.transform(norm_feat_df)

knn.fit(transformed_features_df, y)
y_pred = knn.predict(transformed_features_df)

accuracy = accuracy_score(y, y_pred)
print('Accuracy by PCA:', accuracy)
