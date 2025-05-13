import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0;
        background: linear-gradient(to bottom, #e6f0fa, #ffffff);
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
        padding: 10px 10px 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .stTabs [data-baseweb="tab"] {
        margin-right: 10px;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1f77b4;
    }
    .home-hero {
        position: relative;
        text-align: center;
        color: white;
        padding: 120px 20px;
        background-image: url('https://images.unsplash.com/photo-1682685797660-3d847763208e');
        background-size: cover;
        background-position: center;
        border-radius: 10px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .home-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }
    .home-hero h1 {
        font-size: 4em;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .home-hero p {
        font-size: 1.8em;
        margin: 15px 0 30px;
        position: relative;
        z-index: 1;
    }
    .cta-button {
        display: inline-block;
        padding: 15px 30px;
        background-color: #42a5f5;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        border-radius: 5px;
        text-decoration: none;
        transition: background-color 0.3s;
        margin-top: 20px;
        border: none;
        cursor: pointer;
        pointer-events: auto;
    }
    .cta-button:hover {
        background-color: #1e88e5;
    }
    .logo-header {
        display: flex;
        align-items: center;
        height: 100%;
        padding: 5px 10px;
        background-color: #f0f2f6;
        border-radius: 5px 0 0 5px;
    }
    @media (max-width: 768px) {
        .home-hero {
            padding: 80px 20px;
        }
        .home-hero h1 {
            font-size: 2.8em;
        }
        .home-hero p {
            font-size: 1.4em;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Currency conversion constant
LSL_TO_USD = 0.053  # 1 LSL = 0.053 USD

# Logo and horizontal navigation tabs
logo_col, tabs_col = st.columns([1, 5])
with logo_col:
    st.markdown('<div class="logo-header">', unsafe_allow_html=True)
    logo_path = "images/Motz.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.error(f"Logo file not found at {logo_path}. Please ensure the file exists in the correct directory.")
    st.markdown('</div>', unsafe_allow_html=True)
with tabs_col:
    tab_names = ["Home", "Prediction", "Data Analysis", "Model Evaluation", "Feature Engineering", "Model Training"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('loan_data_set.csv')
    except FileNotFoundError:
        st.error("Dataset 'loan_data_set.csv' not found. Please ensure it is in the working directory.")
        st.stop()
    return data

# Preprocess data for model input
def preprocess_data(data, scaler=None, is_training=False):
    df = data.copy()
    
    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
    
    # Handle missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0] if is_training else 'Male' if col == 'Gender' else 'No')
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median() if is_training else df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0] if is_training else 360)
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0] if is_training else 1)
    
    # Handle Dependents
    df['Dependents'] = df['Dependents'].replace('3+', '3').fillna('0').astype(int)
    
    # Transform categorical variables
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_columns:
        if col == 'Property_Area':
            df[col] = df[col].map({'Urban': 1, 'Rural': 0, 'Semiurban': 0.5})
        else:
            df[col] = df[col].map({
                'Gender': {'Male': 1, 'Female': 0},
                'Married': {'Yes': 1, 'No': 0},
                'Education': {'Graduate': 1, 'Not Graduate': 0},
                'Self_Employed': {'Yes': 1, 'No': 0}
            }[col])
    
    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
    
    # Define feature columns
    feature_columns = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Total_Income', 'Income_Loan_Ratio'
    ]
    
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select features
    features = df[feature_columns]
    
    # Scale features if scaler is provided
    if scaler and not is_training:
        try:
            scaled_features = scaler.transform(features)
            return pd.DataFrame(scaled_features, columns=feature_columns)
        except ValueError as e:
            st.error(f"Scaler error: {str(e)}. The model expects different feature names. Please retrain the model in the Model Training page.")
            st.stop()
    
    return features if not is_training else df

# Home Page
with tab1:
    # Hero Section
    st.markdown("""
        <div class="home-hero">
            <h1>Welcome to the Loan Prediction System</h1>
            <p>Harness the power of machine learning to predict loan approvals with precision.</p>
        </div>
    """, unsafe_allow_html=True)

# Prediction Page
with tab2:
    if not os.path.exists('best_model.joblib') or not os.path.exists('scaler.joblib'):
        st.warning("Model files not found. Please train the model in the Model Training page.")
    else:
        model = joblib.load('best_model.joblib')
        scaler = joblib.load('scaler.joblib')

        st.title("💰 Loan Prediction System")
        st.markdown("""
            This system helps predict loan approval based on various factors.
            Fill in the details below to get a prediction.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            gender = st.selectbox('Gender', ['Male', 'Female'], key='gender')
            married = st.selectbox('Marital Status', ['No', 'Yes'], key='married')
            dependents = st.selectbox('Number of Dependents', ['0', '1', '2', '3+'], key='dependents')
            education = st.selectbox('Education', ['Graduate', 'Not Graduate'], key='education')
            self_employed = st.selectbox('Self Employed', ['No', 'Yes'], key='self_employed')

        with col2:
            st.subheader("Financial Information")
            applicant_income = st.number_input('Monthly Income (LSL)', min_value=0, value=10000, key='applicant_income')
            coapplicant_income = st.number_input('Co-applicant Income (LSL)', min_value=0, value=0, key='coapplicant_income')
            loan_amount = st.number_input('Loan Amount (LSL)', min_value=0, value=200000, key='loan_amount')
            loan_amount_term = st.number_input('Loan Term (Months)', min_value=12, max_value=360, value=360, key='loan_amount_term')
            credit_history = st.selectbox('Credit History', ['Good', 'Bad'], help="Select 'Good' if you have clear credit history", key='credit_history')
            property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'], key='property_area')

        if st.button('Predict Loan Approval'):
            user_data = pd.DataFrame({
                'Gender': [gender],
                'Married': [married],
                'Dependents': [dependents.replace('3+', '3')],
                'Education': [education],
                'Self_Employed': [self_employed],
                'ApplicantIncome': [applicant_income],
                'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount],
                'Loan_Amount_Term': [loan_amount_term],
                'Credit_History': [1 if credit_history == 'Good' else 0],
                'Property_Area': [property_area]
            })

            user_data_scaled = preprocess_data(user_data, scaler=scaler)
            prediction = model.predict(user_data_scaled)
            probability = model.predict_proba(user_data_scaled)[0]

            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction[0] == 1:
                    st.success("✅ Loan Approved!")
                else:
                    st.error("❌ Loan Not Approved")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    title={'text': "Approval Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig)

            with col3:
                st.write("Key Factors:")
                st.write(f"• Monthly Income: LSL {applicant_income:,.2f}")
                st.write(f"• Loan Amount: LSL {loan_amount:,.2f}")
                st.write(f"• Income to Loan Ratio: {(applicant_income + coapplicant_income) / loan_amount:.2f}")
                st.write(f"• Credit History: {credit_history}")
                st.write(f"• Loan Term: {loan_amount_term} months")

        # Add visualization tabs
        st.markdown("---")
        st.subheader("Data Insights")
        tab_insights1, tab_insights2, tab_insights3 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Demographic Analysis"])

        with tab_insights1:
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image('plots/1_loan_status_distribution.png')
                    st.image('plots/6_income_distribution.png')
                except FileNotFoundError:
                    st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")
            with col2:
                try:
                    st.image('plots/8_loan_term_distribution.png')
                    st.image('plots/2_income_vs_loan.png')
                except FileNotFoundError:
                    st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")

        with tab_insights2:
            try:
                st.image('plots/7_correlation_matrix.png')
            except FileNotFoundError:
                st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image('plots/3_loan_by_education.png')
                except FileNotFoundError:
                    st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")
            with col2:
                try:
                    st.image('plots/4_credit_history_impact.png')
                except FileNotFoundError:
                    st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")

        with tab_insights3:
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image('plots/9_gender_analysis.png')
                    st.image('plots/5_property_area_analysis.png')
                except FileNotFoundError:
                    st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")
            with col2:
                try:
                    st.image('plots/10_dependents_analysis.png')
                except FileNotFoundError:
                    st.warning("Plot images not found. Please ensure the 'plots' directory contains the required files.")

# Data Analysis Page
with tab3:
    st.title("📊 Data Analysis")
    st.markdown("Explore the dataset for missing values, outliers, and descriptive statistics.")

    data = load_data()

    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape}")
    st.dataframe(data.head())

    st.subheader("Missing Values")
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_pct})
    st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    if missing.sum() == 0:
        st.write("No missing values.")

    st.subheader("Descriptive Statistics")
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    st.dataframe(data[numerical_features].describe())

    st.subheader("Outlier Detection")
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        return len(outliers)

    outlier_data = {col: detect_outliers(data, col) for col in numerical_features}
    st.write(pd.DataFrame.from_dict(outlier_data, orient='index', columns=['Outliers Detected']))

    st.subheader("Distribution Plots")
    for col in numerical_features:
        fig = px.histogram(data, x=col, title=f"Distribution of {col}", nbins=30)
        st.plotly_chart(fig)

    st.subheader("Categorical Features Distribution")
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_features:
        fig = px.histogram(data, x=col, title=f"Distribution of {col}", histnorm='percent')
        st.plotly_chart(fig)

# Model Evaluation Page
with tab4:
    st.title("📈 Model Evaluation")
    st.markdown("Compare performance of different models.")

    data = load_data()
    processed_data = preprocess_data(data, is_training=True)
    X = processed_data.drop('Loan_Status', axis=1)
    y = processed_data['Loan_Status'].map({'Y': 1, 'N': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    st.subheader("Model Performance")
    results = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            'Model': name,
            'CV Mean Score': cv_scores.mean(),
            'CV Std Score': cv_scores.std(),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score']
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    st.subheader("Performance Visualization")
    fig = px.bar(results_df, x='Model', y=['Test Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                 barmode='group', title="Model Performance Comparison")
    st.plotly_chart(fig)

    st.subheader("Model Advantages")
    for name in models.keys():
        st.write(f"**{name}:**")
        if name == 'Random Forest':
            st.write("- Handles non-linear relationships")
            st.write("- Provides feature importance")
            st.write("- Robust to outliers")
        elif name == 'Logistic Regression':
            st.write("- Simple and interpretable")
            st.write("- Works well with linear relationships")
            st.write("- Less prone to overfitting")
        elif name == 'SVM':
            st.write("- Effective in high-dimensional spaces")
            st.write("- Robust to outliers with appropriate kernel")
        elif name == 'Decision Tree':
            st.write("- Easy to interpret")
            st.write("- Handles non-linear relationships")

# Feature Engineering Page
with tab5:
    st.title("🛠️ Feature Engineering")
    st.markdown("Analyze feature importance and perform feature selection.")

    data = load_data()
    processed_data = preprocess_data(data, is_training=True)
    X = processed_data.drop('Loan_Status', axis=1)
    y = processed_data['Loan_Status'].map({'Y': 1, 'N': 0})

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.dataframe(importance)

    st.subheader("Correlation Matrix")
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Income_Loan_Ratio']
    corr_matrix = X[numerical_features].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Matrix", color_continuous_scale='RdBu')
    st.plotly_chart(fig)

    st.subheader("Feature Selection")
    k = st.slider('Number of Top Features', 1, len(X.columns), 8)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    st.write(f"Selected Top {k} Features: {selected_features}")

# Model Training Page
with tab6:
    st.title("🧠 Model Training")
    st.markdown("Retrain the Random Forest model with custom parameters.")

    data = load_data()
    processed_data = preprocess_data(data, is_training=True)
    X = processed_data.drop('Loan_Status', axis=1)
    y = processed_data['Loan_Status'].map({'Y': 1, 'N': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Training Parameters")
    n_estimators = st.slider("Number of Trees", 50, 200, 100)
    max_depth = st.slider("Max Depth", 5, 30, 10, step=5)
    min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 4, 1)

    if st.button("Train Model"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        st.subheader("Training Results")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        joblib.dump(model, 'best_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        st.success("Model and scaler saved successfully!")

# Footer
st.markdown("---")