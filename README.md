HEAD
# Loan Prediction System

## Project Overview
The Loan Prediction System is designed to predict loan approval based on various applicant details. It uses machine learning models to analyze and predict the likelihood of loan approval.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com//nthola/Loan-Prediction.git
=======
2. Navigate to the project directory: 
   ```bash
   cd loanoredict

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the application
   ```bash
   python app.py

2. Access the application in your web browser at `http://localhost:5000`.

3.Fill in the application details and submit the form.

4.The system will predict the loan approval based on the provides information.

## Project Structure
- `app.py`: Main application script (Streamlit or Flask).
- `train_models.py`: Script for training machine learning models.
- `loan_prediction.py`: Additional script for loan prediction logic.
- `model_evaluation.py`: Script for evaluating model performance.
- `download_data.py`: Script for downloading or preparing data.
- `requirements.txt`: Lists required Python packages.
- `README.md`: Project documentation.
- `Procfile`, `runtime.txt`, `workflow.YAML`: Deployment and workflow files.
- `.gitignore`: Git ignore rules.
- `LICENSE`: Project license.
- `.github/`: GitHub workflows and CI/CD configuration.
  - `workflows/django.yml`: GitHub Actions workflow.
- `.vscode/`: Editor configuration files.
  - `settings.json`: VSCode settings.
- `data/`: Contains datasets (e.g., CSV files).
  - `.gitkeep`: Keeps the directory in version control.
  - `loan_data_set.csv`: Main dataset.
- `models/`: Stores saved machine learning models (e.g., .joblib files).
  - `.gitkeep`: Keeps the directory in version control.
  - `best_model.joblib`: Best performing model.
  - `optimized_best_model.joblib`: Optimized model.
- `plots/`: Contains generated visualizations and analysis plots.
  - `1_loan_status_distribution.png`
  - `2_income_vs_loan.png`
  - `3_loan_by_education.png`
  - `4_credit_history_impact.png`
  - `5_property_area_analysis.png`
  - `6_income_distribution.png`
  - `7_correlation_matrix.png`
  - `8_loan_term_distribution.png`
  - `9_gender_analysis.png`
  - `10_dependents_analysis.png`
- `best_model.joblib`: Best model (duplicate at root, consider cleanup).
- `decision_tree_model.joblib`: Decision tree model.
- `logistic_regression_model.joblib`: Logistic regression model.
- `scaler.joblib`: Feature scaler.
- `svm_model.joblib`: SVM model.
- `label_encoders.joblib`: Label encoders.
- `correlation_matrix.png`: Correlation matrix plot.
- `model_comparison.png`: Model comparison plot.

## Collaborators
- [Nthola](https://github.com//nthola.git)
- [Tsoakae](https://github.com//mimme.git)
- [Limpho](https://github.com//dympho.git)
- [Pheello](https://github.com//coordinator.git)