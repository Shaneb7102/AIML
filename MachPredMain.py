import pandas as pd

# Specify the path to your CSV file
file_path = 'datasets/predictive_maintenance.csv'

# Load only the first 10,000 rows
data = pd.read_csv(file_path, nrows=10000)

# Display the first few rows to confirm it loaded correctly
print(data)