import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model = joblib.load(r"C:\Users\Utsav\OneDrive\Desktop\salary prediction\salary_model (1).pkl")
encoders = joblib.load(r"C:\Users\Utsav\OneDrive\Desktop\salary prediction\encoders (1).pkl")

# Streamlit Page Setup
st.set_page_config(page_title="💼 Salary Predictor", layout="centered")

# App Title
st.title("💰 Employee Salary Prediction")
st.markdown("### 📊 Let's estimate the salary based on employee details!")

# --- Input Form ---
with st.form("salary_form"):
    st.markdown("#### 📝 Fill Employee Information:")

    age = st.slider("🎂 Age", 18, 65, 30)
    gender = st.selectbox("👤 Gender", encoders['Gender'].classes_)
    department = st.selectbox("🏢 Department", encoders['Department'].classes_)
    experience = st.slider("🧠 Experience (in years)", 0, 40, 5)
    performance = st.slider("📈 Performance Score", 1.0, 5.0, 3.0)
    status = st.selectbox("🔘 Employment Status", encoders['Status'].classes_)
    location = st.selectbox("📍 Location", encoders['Location'].classes_)
    session = st.selectbox("⏰ Work Session", encoders['Session'].classes_)

    submitted = st.form_submit_button("🚀 Predict Salary")

# --- Prediction Logic ---
if submitted:
    try:
        # Encode input using saved LabelEncoders
        input_data = {
            'Age': age,
            'Gender': encoders['Gender'].transform([gender])[0],
            'Department': encoders['Department'].transform([department])[0],
            'Performance Score': performance,
            'Experience': experience,
            'Status': encoders['Status'].transform([status])[0],
            'Location': encoders['Location'].transform([location])[0],
            'Session': encoders['Session'].transform([session])[0]
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f"💵 Estimated Salary: ₹{int(prediction):,}")
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
