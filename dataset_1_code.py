import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and preprocess dataset
file_path = 'datasets/dataset_1.csv'
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

# SVM
from sklearn.svm import SVC

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel='rbf', class_weight='balanced', random_state=42):
    print("SVM Model:\n")
    svm_model = SVC(kernel=kernel, class_weight=class_weight, random_state=random_state)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

train_and_evaluate_svm(X_train, y_train, X_test, y_test)

# ANN
from sklearn.neural_network import MLPClassifier

def train_and_evaluate_ann(X_train, y_train, X_test, y_test, solver='adam', alpha=1e-5, random_state=42):
    print("\nANN Model:\n")
    ann_model = MLPClassifier(solver=solver, alpha=alpha, random_state=random_state)
    ann_model.fit(X_train, y_train)
    y_pred = ann_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

train_and_evaluate_ann(X_train, y_train, X_test, y_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    print("\nRandom Forest Model:\n")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

def train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    print("\nNaive Bayes Model:\n")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test)

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    print("\nK-Nearest Neighbors (KNN) Model:\n")
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

train_and_evaluate_knn(X_train, y_train, X_test, y_test)
