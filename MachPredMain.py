import pandas as pd
import sklearn

# Specify the path to your CSV file
file_path = 'datasets/predictive_maintenance.csv'

# Load only the first 10,000 rows
data = pd.read_csv(file_path, nrows=10000)


data.drop(columns=['UDI'], inplace=True)

from sklearn.preprocessing import LabelEncoder

if 'Failure Type' in data.columns:
    le = LabelEncoder()
    data['Failure Type'] = le.fit_transform(data['Failure Type'])
    print("Label Encoded Failure Types:\n", data[['Failure Type']].head())


#drop rows with empty values
data_cleaned = data.dropna()


# Display the first few rows to confirm it loaded correctly
print(data_cleaned)