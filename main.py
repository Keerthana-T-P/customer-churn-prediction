import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


os.environ['LOKY_MAX_CPU_COUNT'] = '4'


data = pd.read_csv('C:/Users/tpgop/OneDrive/Desktop/me/customer-churn-prediction/Churn-Data.csv')  # Make sure the file name and path are correct

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

le = LabelEncoder()
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'TV_Streaming', 
                        'Movie_Streaming', 'Contract', 'PaperlessBilling', 'Method_Payment', 'Churn']
for feature in categorical_features:
    data[feature] = le.fit_transform(data[feature])

scaler = StandardScaler()
numerical_features = ['tenure', 'Charges_Month', 'TotalCharges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

X = data.drop(columns=['cID', 'Churn'])
y = data['Churn']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print(f"Best Model Accuracy: {accuracy_best:.4f}")
print(f"Best Model F1 Score: {f1_best:.4f}")
