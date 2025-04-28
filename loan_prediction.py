import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.model_selection import cross_val_score

# Load dataset
file_path = 'loan_data_set.csv'
data = pd.read_csv(file_path)

# Data Cleaning
imputer = SimpleImputer(strategy='mean')
data['LoanAmount'] = imputer.fit_transform(data[['LoanAmount']])
data['Loan_Amount_Term'] = imputer.fit_transform(data[['Loan_Amount_Term']])
data['Credit_History'] = imputer.fit_transform(data[['Credit_History']])

# Encoding categorical features
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'].fillna('Male'))
data['Married'] = label_encoder.fit_transform(data['Married'].fillna('No'))
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Self_Employed'] = label_encoder.fit_transform(data['Self_Employed'].fillna('No'))
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

# Add feature engineering
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Income_Loan_Ratio'] = data['Total_Income'] / data['LoanAmount']

# Feature extraction
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Development
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Model evaluation with cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f'\nCross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print('\nFeature Importance:')
print(feature_importance)

# Save the model and preprocessing objects
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')