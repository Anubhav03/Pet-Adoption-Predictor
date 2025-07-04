import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load dataset
dataset = pd.read_csv('pet_adoption_data.csv')

# Separate features and target
X = dataset.iloc[:, 0:-1] 
y = dataset.iloc[:, -1]    

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X)

# Save the list of columns for future use
model_columns = X_encoded.columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

# Train the model
classifier = SVC(kernel='rbf', random_state=0, probability=True)
classifier.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(classifier, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
