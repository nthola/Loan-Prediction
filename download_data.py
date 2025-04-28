import pandas as pd
import os

# Set the file path relative to the current script location
file_path = os.path.join(os.path.dirname(__file__), 'loan_data_set.csv')

# Check if file exists and load it
if os.path.exists(file_path):
    print(f"Using existing dataset from {file_path}")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(f"Total records: {len(df)}")
else:
    print("Error: Dataset not found in the project directory!")