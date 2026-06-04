import streamlit as st 
import pandas as pd
import joblib

model=joblib.load("Logistic_Regression_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    
    /* Subtitle */
    .subtitle {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    
    /* Card container for inputs */
    .stDataFrame, .stButton > button {
        border-radius: 12px;
    }
    
    /* Header for input section */
    .input-header {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .input-header h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(238, 82, 83, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(238, 82, 83, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Success and error messages */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1);
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .stSuccess strong {
        color: #28a745;
        font-size: 1.3rem;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1);
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .stError strong {
        color: #dc3545;
        font-size: 1.3rem;
    }
    
    /* Input labels */
    label {
        font-weight: 600;
        color: #333;
    }
    
    /* Selectbox and slider styling */
    .stSlider > div > div > div {
        border-radius: 8px;
    }
    
    /* Info box */
    .info-box {
        background-color: rgba(255,255,255,0.15);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("❤️ Heart Disease Prediction")
    st.markdown('<p class="subtitle">Advanced ML-powered risk assessment by AARAV</p>', unsafe_allow_html=True)

# Info section
st.markdown("""
<div class="info-box">
    <strong>ℹ️</strong> Enter patient's health parameters below. The model will analyze the data and provide a risk assessment.
</div>
""", unsafe_allow_html=True)

# Input section with organized layout
st.markdown('<div class="input-header"><h3>📋 Patient Health Information</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 100, 40, help="Patient's age in years")
    sex = st.selectbox("👤 Sex", ["M", "F"], help="Patient's biological sex")
    chest_pain = st.selectbox("胸口 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="Type of chest pain experienced")
    resting_bp = st.number_input("❤️‍🩹 Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure when heart is at rest")
    cholesterol = st.number_input("💊 Cholesterol (mg/dL)", 100, 600, 200, help="Total cholesterol level")
    fasting_bs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dL", [0, 1], help="1 if fasting blood sugar is high, 0 otherwise")

with col2:
    resting_ecg = st.selectbox(" heartbeat Resting ECG", ["Normal", "ST", "LVH"], help="Resting electrocardiogram result")
    max_hr = st.slider("🏃 Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
    exercise_angina = st.selectbox("💪 Exercise-Induced Angina", ["Y", "N"], help="1 if angina triggered by exercise, 0 otherwise")
    oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1, help="ST depression induced by exercise")
    st_slope = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment")

# Prediction button
if st.button("🔮 Predict Heart Disease Risk"):
    with st.spinner("Analyzing patient data..."):
        # Prepare input data
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale only numerical columns (scaler was fitted on these 5 columns)
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        input_df_scaled = input_df.copy()
        input_df_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols].values)

        # Make prediction
        prediction = model.predict(input_df_scaled)[0]
        proba = model.predict_proba(input_df_scaled)[0]


    # Display result
    if prediction == 1:
        st.markdown("""
        <div style="background-color: rgba(220, 53, 69, 0.1); border-left: 5px solid #dc3545; padding: 25px; border-radius: 12px; margin-top: 20px; text-align: center;">
            <h2 style="color: #dc3545;">⚠️ High Risk of Heart Disease</h2>
            <p style="color: #666; font-size: 1.1rem;">Please consult a cardiologist for further evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: rgba(40, 167, 69, 0.1); border-left: 5px solid #28a745; padding: 25px; border-radius: 12px; margin-top: 20px; text-align: center;">
            <h2 style="color: #28a745;">✅ Low Risk of Heart Disease</h2>
            <p style="color: #666; font-size: 1.1rem;">Maintain a healthy lifestyle to keep your heart healthy!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a divider and info
    st.markdown("---")
    st.info("💡 Note: This is an AI-powered prediction tool. Please consult with a healthcare professional for medical advice.")
