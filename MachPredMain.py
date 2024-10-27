import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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
    print("Label Encoded Failure Types:\n", data['Failure Type'].head())

# Encode 'Type' column if it exists in the data
if 'Type' in data.columns:
    data['Type'] = le.fit_transform(data['Type'])

# Drop rows with empty values
data_cleaned = data.dropna()

# Standardize numerical columns
# Identify numerical columns (excluding 'Failure Type')
numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns.drop('Failure Type')

# Initialize the scaler and apply it to the numerical columns
scaler = StandardScaler()
data_cleaned[numerical_columns] = scaler.fit_transform(data_cleaned[numerical_columns])

# Separate features and target
X = data_cleaned.drop(columns=['Failure Type'])
y = data_cleaned['Failure Type']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split the balanced data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Display the first few rows of the preprocessed training data
print("Training Data:\n", X_train.head())
print("Training Labels Distribution:\n", y_train.value_counts())