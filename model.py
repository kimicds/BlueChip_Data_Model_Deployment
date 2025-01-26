import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
with open("rdf.pkl", "rb") as file:
    model = joblib.load(file)

# Define the app layout and navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home Page", "User Entry", "Model Prediction"])

# Define training columns
training_columns = [['Gender','Married','Education',
 'Self_Employed','ApplicantIncome','CoapplicantIncome',
 'LoanAmount','Loan_Amount_Term','Credit_History','Total_Income',
 'Dependents_1','Dependents_2','Dependents_3',
 'Property_Area_1','Property_Area_2']
    
]

# Function for log transformation
def apply_log_transformation(input_data):
    input_data['ApplicantIncome'] = np.log1p(input_data['ApplicantIncome'])
    input_data['CoapplicantIncome'] = np.log1p(input_data['CoapplicantIncome'])
    input_data['LoanAmount'] = np.log1p(input_data['LoanAmount'])
    input_data['Total_Income'] = np.log1p(input_data['Total_Income'])
    return input_data

# Initialize the StandardScaler
scaler = StandardScaler()

# Home Page
if page == "Home Page":
    st.title("Loan Creditworthiness Prediction")
    st.write("""
        Welcome to the Loan Prediction Web App! 
        Use the sidebar to navigate to the User Entry page to provide input data
        or the Model Prediction page to see the prediction.
    """)


# User Entry
elif page == "User Entry":
    st.title("User Data Entry")
    
    # Split the layout into two columns
    col1, col2 = st.columns(2)
    
    # Categorical inputs in the first column
    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Education = st.radio("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Property_Area = st.selectbox("Property Area", ["Rural", "Urban", "Semiurban"])
    
    # Numerical inputs in the second column
    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value= 100)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
        LoanAmount = st.number_input("Loan Amount", min_value= 17)
        Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value= 200)
        Credit_History = st.selectbox("Credit History", [1, 0])
        Total_Income = st.number_input("Total Income", min_value= 1000)  



    # Save user input to a session state
    if st.button("Save Data"):
        st.session_state.user_input = {
            "Gender": Gender,
            "Married": Married,
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Total_Income": Total_Income,
            "Dependents": Dependents,
            "Property_Area": Property_Area,
        }
        st.success("Data saved! Navigate to the Model Prediction page to make predictions.")

# Model Prediction
elif page == "Model Prediction":
    st.title("Model Prediction")
    
    if "user_input" not in st.session_state:
        st.warning("No data found! Please go to the User Entry page to provide input.")
    else:
        # Retrieve saved user input
        user_input = st.session_state.user_input

        # Display user input
        st.subheader("User Input Data")
        for key, value in user_input.items():
            st.write(f"**{key}:** {value}")


        # Button to confirm and predict
        if st.button("Make Prediction"):
             
            # Convert user input to DataFrame
            input_df = pd.DataFrame([user_input])

            # Apply log transformation on relevant features
            input_df = apply_log_transformation(input_df)

            # Map Dependents and Property_Area to numeric
            input_df["Dependents"] = input_df["Dependents"].replace({"0": "0", "1": "1", "2": "2", "3+": "3"})
            input_df["Property_Area"] = input_df["Property_Area"].replace({"Urban": "1", "Semiurban": "2", "Rural": "0"})

            # One-hot encode the data (drop_first=True for consistency with training)
            input_df = pd.get_dummies(input_df, columns=["Dependents", "Property_Area"], drop_first=True)

            # Align the DataFrame with the training columns
            input_df = input_df.reindex(columns=training_columns, fill_value=0)

            # Convert to numpy array for prediction
            input_array = input_df.to_numpy()

            # Make prediction
            try:
                prediction = model.predict(input_array)
         

                st.write(f"**Prediction:** {'Loan Approved' if prediction[0] == 1 else 'Loan Rejected'}")
            
            except Exception as e:
                st.error(f"Error during prediction: {e}")
