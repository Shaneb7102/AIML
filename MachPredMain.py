import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Specify the path to your CSV file
file_path = 'datasets/predictive_maintenance.csv'

# Load only the first 10,000 rows
data = pd.read_csv(file_path, nrows=10000)

# Drop unnecessary columns
data.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode categorical columns if they exist
le = LabelEncoder()

# Encode 'Failure Type' if it's present in the data
if 'Failure Type' in data.columns:
    data['Failure Type'] = le.fit_transform(data['Failure Type'])
    print("Label Encoded Failure Types:\n", data[['Failure Type']])


# Encode 'Type' column if it exists in the data
if 'Type' in data.columns:
    data['Type'] = le.fit_transform(data['Type'])

# Drop rows with empty values
data_cleaned = data.dropna()

# Display the first few rows of the cleaned dataset
print("Cleaned Data:\n", data_cleaned.head())