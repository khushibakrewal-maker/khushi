
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Product Prediction",
    page_icon="🏨",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL FROM HUGGING FACE MODEL HUB
# --------------------------------------------------

MODEL_REPO = "Khushibk/tourism-product-prediction"
MODEL_FILE = "tourism_best_model.pkl"


@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    model = joblib.load(model_path)

    return model


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏨 Tourism Product Purchase Prediction")

st.write(
    "Enter customer details to predict whether "
    "the customer is likely to purchase the tourism product."
)


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=1,
    max_value=10,
    value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Divorced",
        "Unmarried",
        "Single"
    ]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=1,
    value=2
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)


# --------------------------------------------------
# CREATE DATAFRAME AND PREDICT
# --------------------------------------------------

if st.button("Predict Purchase"):

    input_data = {
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    }

    # Convert user input into DataFrame
    input_df = pd.DataFrame(input_data)

    st.subheader("Customer Input")
    st.dataframe(input_df)

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:

        st.success(
            "Customer is likely to purchase the tourism product."
        )

    else:

        st.warning(
            "Customer is unlikely to purchase the tourism product."
        )

    st.write(
        f"Purchase Probability: **{probability:.2%}**"
    )
