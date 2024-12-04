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
from sklean.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# from keras.models import Sequential
# from keras.layers import Dense, Input

dataset = pd.read_csv("diabeties_cluster.csv")

X = dataset[['Glucose', 'BMI', 'Age']]
y = dataset['Outcome']
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


#step 4 : Naive Bayes

#Create Naive Bayes Model
nb_model = GaussianNB() # CategoricalNB() for categorical data
nb_model.fit(train_x, train_y)
#Accuracy of Model
print("Test accuracy: ", nb_model.score(test_x,test_y))

#Create NN Model
# nn = Sequential()
# nn = MLPClassifier(max_iter=1000, random_state=42)
# nn.add(Input(shape=(8,)))

# Define the MLPClassifier (equivalent of 1 hidden layer with 5 neurons)
nn = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Train the model
nn.fit(train_x, train_y)

# Evaluate the model on test data
accuracy = nn.score(test_x, test_y)
print(f'Accuracy: {accuracy:.4f}')


# KNN
scaler = StandardScaler()
train_x_knn_scaled = scaler.fit_transform(train_x)
test_x_knn_scaled = scaler.transform(test_x)

accuracies = []
# No r-squared value

# range is 1 to 19
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x_knn_scaled, train_y)
    y_pred = knn.predict(test_x_knn_scaled)
    accuracy = accuracy_score(test_y, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for k = {k}: {accuracy:.4f}")

# Reporting the best K
best_k = accuracies.index(max(accuracies)) + 1
best_accuracy = max(accuracies)

print(f"Best K: {best_k}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Ploting the accuracies - This is the optional step

plt.plot(range(1, 20), accuracies)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. K')
plt.show()

