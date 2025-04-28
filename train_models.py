import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load the dataset
print("Loading dataset...")
try:
    # Try different possible filenames
    possible_filenames = ['loan_data_set.csv', 'loan_prediction.csv', 'loan.csv', 'train.csv']
    
    for filename in possible_filenames:
        try:
            df = pd.read_csv(filename)
            print(f"Successfully loaded {filename}")
            break
        except FileNotFoundError:
            continue
    else:  # This runs if the loop completes without a break
        raise FileNotFoundError("No loan dataset found")
        
except FileNotFoundError:
    print("Error: No loan dataset file found. Please download the loan prediction dataset and save it as 'loan_data.csv'.")
    exit()

# Preprocess the data
print("Preprocessing data...")

# Check if Loan_ID column exists and drop it
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)
    print("Dropped Loan_ID column")

# Handle missing values using the recommended pandas approach
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    df[col] = df[col].fillna(df[col].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Convert categorical to numerical
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = df['Dependents'].astype(int)

# Create additional features
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']

# Print column names and data types to debug
print("\nDataset columns:")
print(df.dtypes)

# Encode categorical variables
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_columns:
    df[col] = df[col].map({'Male': 1, 'Female': 0, 
                           'Yes': 1, 'No': 0, 
                           'Graduate': 1, 'Not Graduate': 0,
                           'Urban': 1, 'Rural': 0, 'Semiurban': 0.5})

# Prepare features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Check for any remaining non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) > 0:
    print(f"\nWarning: Found non-numeric columns: {non_numeric_cols}")
    print("Dropping these columns...")
    X = X.drop(non_numeric_cols, axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved as 'scaler.joblib'")

# Train and save models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Save the model
    if name == 'Random Forest':
        joblib.dump(model, 'best_model.joblib')
        print(f"{name} model saved as 'best_model.joblib'")
    else:
        filename = f"{name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(model, filename)
        print(f"{name} model saved as '{filename}'")

print("\nAll models have been trained and saved successfully!")


# Handle outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df

# Apply outlier removal to relevant columns
df = remove_outliers(df, 'ApplicantIncome')
df = remove_outliers(df, 'CoapplicantIncome')
df = remove_outliers(df, 'LoanAmount')


# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Perform Grid Search
print("\nPerforming Grid Search for Random Forest...")
grid_search.fit(X_train_scaled, y_train)

# Get best parameters and model
best_rf = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the best model
y_pred = best_rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Random Forest Accuracy: {accuracy:.4f}")

# Save the optimized model
joblib.dump(best_rf, 'optimized_best_model.joblib')
print("Optimized Random Forest model saved as 'optimized_best_model.joblib'")