import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

file_path = 'datasets/predictive_maintenance.csv'
data = pd.read_csv(file_path)


data.drop(columns=['UDI', 'Product ID'], inplace=True, errors='ignore')


data['Failure'] = data['Failure Type'].apply(lambda x: 0 if x == 'No Failure' else 1)

data.drop(columns=['Failure Type'], inplace=True)

data = pd.get_dummies(data, columns=['Type'], drop_first=True)


numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.drop('Failure')
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data.drop(columns=['Failure'])
y = data['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Sample:\n", X_train.head())
print("Training Labels Distribution:\n", y_train.value_counts())


#SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel='rbf', class_weight='balanced', random_state=42):
    
    # Step 1: Initialize the SVM model
    svm_model = SVC(kernel=kernel, class_weight=class_weight, random_state=random_state)
    
    # Step 2: Train the model
    svm_model.fit(X_train, y_train)
    
    # Step 3: Make predictions
    y_pred = svm_model.predict(X_test)
    
    # Step 4: Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


train_and_evaluate_svm(X_train, y_train, X_test, y_test)
#########################################################

#ANN