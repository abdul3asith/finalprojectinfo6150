import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier

# Sample dataset (replace with your actual dataset)
dataset = pd.read_csv("diabeties_cluster.csv")

X = dataset[['Glucose', 'BMI', 'Age']]
y = dataset['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
nb = GaussianNB()
nn = MLPClassifier(max_iter=1000, random_state=42)
knn = KNeighborsClassifier()

# Define meta-learner (decision tree)
meta_learner = DecisionTreeClassifier(random_state=42)

# Hyperparameter grids for base classifiers
param_grid_nb = {}  # No hyperparameters to tune for GaussianNB

param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

param_grid_meta = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search for base classifiers
gs_nn = GridSearchCV(nn, param_grid_nn, cv=5)
gs_nn.fit(X_train, y_train)

gs_knn = GridSearchCV(knn, param_grid_knn, cv=5)
gs_knn.fit(X_train, y_train)

# Create a stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('nb', nb),
        ('nn', gs_nn.best_estimator_),
        ('knn', gs_knn.best_estimator_)
    ],
    final_estimator=GridSearchCV(meta_learner, param_grid_meta, cv=5),
    cv=5
)

# Train the stacking classifier
stacking_clf.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Results
print(f"Best Neural Network params: {gs_nn.best_params_}")
print(f"Best KNN params: {gs_knn.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")
