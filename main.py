import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing

dataset = pd.read_csv("Loss_of_Customers.csv")
# print(dataset.head())

# removing outliers using Interquartile range (IQR)
def removing_outliers(dataset, columns):
    for col in columns:
        q3, q1 = np.percentile(dataset[col], [75, 25])
        fence = 1.5 * (q3 - q1)
        upper_band = q3 + fence
        lower_band = q1 - fence
        dataset = dataset[(dataset[col] >= lower_band) & (dataset[col] <= upper_band)]

    return dataset

new1_dataset = removing_outliers(dataset, ['balance', 'estimated_salary', 'products_number', 'tenure', 'age', 'active_member', 'credit_card',  'credit_score'])
# print(dataset)

print(new1_dataset)

# encoding
#label encoding for sex column.
label_encoder = preprocessing.LabelEncoder()
new1_dataset['gender'] = label_encoder.fit_transform(new1_dataset['gender'])
print(new1_dataset['gender'])
#one hot encoding for country column
data_encoded = pd.get_dummies(new1_dataset, columns=["country"], drop_first=True)
print(data_encoded.head())


imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(data_encoded)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=data_encoded.columns)
print(imputed_dataframe)

imputed_dataframe['gender'] = imputed_dataframe['gender'].astype(int)
imputed_dataframe['country_Germany'] = imputed_dataframe['country_Germany'].astype(int)
imputed_dataframe['country_Spain'] = imputed_dataframe['country_Spain'].astype(int)
print(imputed_dataframe[['gender', 'country_Spain', 'country_Germany']])

# imputed_dataframe.to_csv("Loss_of_Customers_Cleaned.csv", index=False)

print(imputed_dataframe)
#normalise data
norm_dataset = (imputed_dataframe - imputed_dataframe.mean()) / imputed_dataframe.std()
print(norm_dataset)

norm_dataset.to_csv("Loss_of_Customers_norm.csv",index=False)

