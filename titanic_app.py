import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('titanic_model.pkl')

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# --- Input Fields ---
# 1. Passenger Class (The missing option!)
pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))

# 2. Gender
sex = st.radio("Gender", ["Male", "Female"])

# 3. Age
age = st.slider("Age", 0, 100, 30)

# 4. Family Members
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)

# 5. Fare & Port
fare = st.number_input("Fare Price (£)", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# --- Logic to Convert Inputs to Numbers ---
sex_num = 0 if sex == "Male" else 1

embarked_Q = 1 if "Q" in embarked else 0
embarked_S = 1 if "S" in embarked else 0

# --- Prediction ---
if st.button("Predict Survival"):
    input_data = pd.DataFrame([[pclass, sex_num, age, sibsp, parch, fare, embarked_Q, embarked_S]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Result: The passenger likely SURVIVED! 🎉")
    else:
        st.error("Result: The passenger likely DIED. 💀")
