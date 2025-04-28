import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Try to load the model and scaler, if not found, create new ones
try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run the model training script first.")
    st.stop()

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
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💰 Loan Prediction System")
st.markdown("""
    This system helps predict loan approval based on various factors.
    Fill in the details below to get a prediction.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Marital Status', ['No', 'Yes'])
    dependents = st.selectbox('Number of Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['No', 'Yes'])

# Add currency conversion constant
LSL_TO_USD = 0.053  # 1 LSL = 0.053 USD (approximate rate)

with col2:
    st.subheader("Financial Information")
    
    # Remove validation message and handle invalid input
    applicant_income = st.number_input('Monthly Income (LSL)', min_value=0, value=10000)
    coapplicant_income = st.number_input('Co-applicant Income (LSL)', min_value=0, value=0)
    loan_amount = st.number_input('Loan Amount (LSL)', min_value=0, value=200000)
    loan_amount_term = st.number_input('Loan Term (Months)', min_value=12, max_value=360, value=360)
    
    credit_history = st.selectbox('Credit History', ['Good', 'Bad'], help="Select 'Good' if you have clear credit history")
    property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

# Add a predict button
if st.button('Predict Loan Approval'):
    # Prepare the input data
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

    # Calculate additional features
    user_data['Total_Income'] = user_data['ApplicantIncome'] + user_data['CoapplicantIncome']
    user_data['Income_Loan_Ratio'] = user_data['Total_Income'] / user_data['LoanAmount']

    # Transform categorical variables (matching the training data encoding)
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_columns:
        user_data[col] = 1 if user_data[col].iloc[0] in ['Male', 'Yes', 'Graduate', 'Yes', 'Urban'] else 0

    # Scale the features
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)[0]

    # Display result with custom styling
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction[0] == 1:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Not Approved")
    
    with col2:
        # Create a gauge chart for approval probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability[1] * 100,
            title = {'text': "Approval Probability"},
            gauge = {
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
        # Display key factors with LSL
        st.write("Key Factors:")
        st.write(f"• Monthly Income: LSL {applicant_income:,.2f}")
        st.write(f"• Loan Amount: LSL {loan_amount:,.2f}")
        st.write(f"• Income to Loan Ratio: {user_data['Income_Loan_Ratio'].iloc[0]:.2f}")
        st.write(f"• Credit History: {credit_history}")
        st.write(f"• Loan Term: {loan_amount_term} months")

# Add footer
# Add after the prediction section:

# Add visualization tabs
st.markdown("---")
st.subheader("Data Insights")
tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Demographic Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.image('plots/1_loan_status_distribution.png')
        st.image('plots/6_income_distribution.png')
    with col2:
        st.image('plots/8_loan_term_distribution.png')
        st.image('plots/2_income_vs_loan.png')

with tab2:
    st.image('plots/7_correlation_matrix.png')
    col1, col2 = st.columns(2)
    with col1:
        st.image('plots/3_loan_by_education.png')
    with col2:
        st.image('plots/4_credit_history_impact.png')

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.image('plots/9_gender_analysis.png')
        st.image('plots/5_property_area_analysis.png')
    with col2:
        st.image('plots/10_dependents_analysis.png')
        