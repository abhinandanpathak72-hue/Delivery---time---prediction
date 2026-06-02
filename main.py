import streamlit as st
import pandas as pd
import pickle

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# ----------------------------------
# Load Model and Encoders
# ----------------------------------
@st.cache_resource
def load_files():
    with open("random_forest_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("label_encoders.pkl", "rb") as encoder_file:
        encoders = pickle.load(encoder_file)

    return model, encoders


model, encoders = load_files()

# ----------------------------------
# Custom Styling
# ----------------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #0e1117;
    border: 2px solid #00c853;
    text-align: center;
    margin-top: 20px;
}

.prediction-value {
    font-size: 40px;
    font-weight: bold;
    color: #00c853;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Header
# ----------------------------------
st.title("🚚 Food Delivery Time Prediction")
st.write(
    "Predict estimated food delivery time using a trained Random Forest Model."
)

st.divider()

# ----------------------------------
# Input Form
# ----------------------------------
col1, col2 = st.columns(2)

with col1:

    distance = st.number_input(
        "📍 Distance (km)",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.1
    )

    weather = st.selectbox(
        "🌦 Weather",
        list(encoders["Weather"].classes_)
    )

    traffic = st.selectbox(
        "🚦 Traffic Level",
        list(encoders["Traffic_Level"].classes_)
    )

with col2:

    time_of_day = st.selectbox(
        "🕒 Time Of Day",
        list(encoders["Time_of_Day"].classes_)
    )

    vehicle = st.selectbox(
        "🏍 Vehicle Type",
        list(encoders["Vehicle_Type"].classes_)
    )

    prep_time = st.number_input(
        "🍽 Preparation Time (minutes)",
        min_value=1,
        max_value=120,
        value=15
    )

    experience = st.number_input(
        "👨‍💼 Courier Experience (years)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5
    )

st.divider()

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔮 Predict Delivery Time"):

    try:

        input_data = pd.DataFrame({
            "Distance_km": [distance],
            "Weather": [weather],
            "Traffic_Level": [traffic],
            "Time_of_Day": [time_of_day],
            "Vehicle_Type": [vehicle],
            "Preparation_Time_min": [prep_time],
            "Courier_Experience_yrs": [experience]
        })

        # Encode categorical columns
        input_data["Weather"] = encoders["Weather"].transform(
            input_data["Weather"]
        )

        input_data["Traffic_Level"] = encoders["Traffic_Level"].transform(
            input_data["Traffic_Level"]
        )

        input_data["Time_of_Day"] = encoders["Time_of_Day"].transform(
            input_data["Time_of_Day"]
        )

        input_data["Vehicle_Type"] = encoders["Vehicle_Type"].transform(
            input_data["Vehicle_Type"]
        )

        # Prediction
        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>Estimated Delivery Time</h2>
                <div class="prediction-value">
                    {prediction:.1f} Minutes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Random Forest Regressor • Food Delivery Time Prediction")