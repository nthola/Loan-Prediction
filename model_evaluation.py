# Add joblib to imports at the top
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import joblib  # Add this import

# Load and preprocess data (similar to loan_prediction.py)
# Load data
file_path = 'loan_data_set.csv'
data = pd.read_csv(file_path)

# TASK 1a: Missing Values Analysis
print("\nMissing Values Analysis:")
print(data.isnull().sum())
print("\nMissing Values Percentage:")
print((data.isnull().sum() / len(data)) * 100)

# TASK 1b: Outliers Detection
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers)

numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
print("\nOutliers Analysis:")
for column in numerical_features:
    outliers_count = detect_outliers(data, column)
    print(f"{column}: {outliers_count} outliers detected")

# Define categorical features before using them
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

# TASK 1c: Descriptive Analysis
print("\nDescriptive Statistics:")
print(data[numerical_features].describe())
print("\nCategorical Features Distribution:")
for column in categorical_features:
    print(f"\n{column} Distribution:")
    print(data[column].value_counts(normalize=True) * 100)

# Handle missing values BEFORE feature engineering
# Numerical missing values
numerical_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Categorical missing values
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Handle Dependents column separately
data['Dependents'] = data['Dependents'].fillna('0')
data['Dependents'] = data['Dependents'].replace('3+', '3')
data['Dependents'] = data['Dependents'].astype(float)

# Verify missing values after filling
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())
print("\nMissing Values Percentage After Cleaning:")
print((data.isnull().sum() / len(data)) * 100)

# Continue with encoding
# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

data['Loan_Status'] = LabelEncoder().fit_transform(data['Loan_Status'])

# Feature engineering
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Income_Loan_Ratio'] = data['Total_Income'] / data['LoanAmount']

# Prepare features
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Evaluate models
results = []
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    
    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Store results
    results.append({
        'Model': name,
        'CV Mean Score': cv_scores.mean(),
        'CV Std Score': cv_scores.std(),
        'Test Accuracy': accuracy_score(y_test, y_pred)
    })
    
    print(f'\nResults for {name}:')
    print(f'Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

# Plot results
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Test Accuracy', data=results_df)
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Save best model
best_model = models[results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']]

# Task 2a: Enhanced Feature Engineering
print("\nFeature Selection Analysis:")
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)
print(feature_scores)

# Correlation Analysis
plt.figure(figsize=(12, 8))
correlation_matrix = data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Task 2b: Feature Tuning
# Select top K features
k_best_features = 8
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"\nSelected top {k_best_features} features:")
print(selected_features)

# Task 3: Model Determination with Reasoning
print("\nModel Selection Reasoning:")
for name, model in models.items():
    print(f"\n{name}:")
    print("Advantages:")
    if name == 'Random Forest':
        print("- Handles non-linear relationships")
        print("- Provides feature importance")
        print("- Robust to outliers")
    elif name == 'Logistic Regression':
        print("- Simple and interpretable")
        print("- Works well with linear relationships")
        print("- Less prone to overfitting")
    