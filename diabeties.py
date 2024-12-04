import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing

dataset = pd.read_csv("diabetes_project.csv")
print(dataset.head())

#Removing Outliers
def remove_outliers(dataset, columns):
    for col in columns:
        if dataset[col].notna().sum() > 0:
            q3, q1 = np.percentile(dataset[col].dropna(), [75, 25])
            fence = 1.5 * (q3 - q1)
            upper_band = q3 + fence
            lower_band = q1 - fence

            print(f'Column: {col}')
            print(f'q1 = {q1}, q3 = {q3}, IQR = {q3 - q1}, upper = {upper_band}, lower = {lower_band}')

            # Replace outliers with None (or np.nan)
            dataset.loc[(dataset[col] < lower_band) |
                                (dataset[col] > upper_band), col] = None

    return dataset

new1_dataset = remove_outliers(dataset, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
print(new1_dataset)

#Imputing missing values
new1_dataset['Glucose'] = new1_dataset['Glucose'].fillna(new1_dataset['Glucose'].mean())
new1_dataset['Age'] = new1_dataset['Age'].fillna(new1_dataset['Age'].mean())
new1_dataset['BMI'] = new1_dataset['BMI'].fillna(new1_dataset['BMI'].mean())
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(new1_dataset)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=new1_dataset.columns)
print(imputed_dataframe.round(2))

imputed_dataframe.to_csv('diabeties_imputed.csv', index=False)

#normalise data
norm_dataset = (imputed_dataframe - imputed_dataframe.mean()) / imputed_dataframe.std()
print(norm_dataset)

norm_dataset.to_csv('diabeties_normalised.csv', index=False)

# print(norm_dataset.columns)

# Step 2 ---------------------------------------------------------------------------------------------------------
# Clustering




