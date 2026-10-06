import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SugarSense",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MEDICAL_THRESHOLD = 0.240

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# ============================================================
# MODEL TRANSFORMER
# Required for loading the saved .joblib model
# ============================================================

class ZeroToNaN(BaseEstimator, TransformerMixin):

    def __init__(self, zero_cols=None):
        self.zero_cols = zero_cols or ZERO_AS_MISSING

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        for col in self.zero_cols:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

        return X


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    x = df.copy()

    # Missing-value indicators
    for col in ZERO_AS_MISSING:
        x[f"{col}_Missing"] = (x[col] == 0).astype(int)

    # Obesity flag
    x["ObesityFlag"] = (x["BMI"] >= 30).astype(int)

    # Interaction features
    x["Glucose_BMI"] = x["Glucose"] * x["BMI"]

    x["Age_BMI"] = x["Age"] * x["BMI"]

    return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load("sugarsense_svm.joblib")


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🩺 SugarSense")

st.subheader("Early Diabetes Risk Screening")

st.write(
    "Enter the patient's clinical measurements below. "
    "SugarSense estimates diabetes risk using a supervised "
    "machine-learning model."
)

st.warning(
    "⚠️ This tool is for screening purposes only. "
    "It does not provide a medical diagnosis."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🩺 SugarSense")

    st.write("### About the Model")

    st.write(
        "SugarSense uses a trained Support Vector Machine "
        "(SVM) classification model."
    )

    st.divider()

    st.write("### Screening Threshold")

    st.metric(
        "Screening Threshold",
        f"{MEDICAL_THRESHOLD * 100:.1f}%"
    )

    st.caption(
        f"Model threshold: {MEDICAL_THRESHOLD:.3f}"
    )

    st.caption(
        "The threshold is optimized for a screening-oriented "
        "use case where identifying potential positive cases "
        "is important."
    )

    st.divider()

    st.write("### Important")

    st.caption(
        "This application provides an estimated risk score. "
        "It does not diagnose diabetes."
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

st.write(
    "Please enter the patient's clinical measurements."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("patient_form"):

    left, right = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with left:

        st.subheader("Clinical Measurements")

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            step=1.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=72.0,
            step=1.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with right:

        st.subheader("Additional Measurements")

        insulin = st.number_input(
            "Insulin (µU/ml)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0
        )

        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.50,
            step=0.01
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    st.write("")

    submitted = st.form_submit_button(
        "🔍 Check Diabetes Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # CREATE PATIENT DATAFRAME
    # --------------------------------------------------------

    patient = pd.DataFrame([{

        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age

    }])


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    patient_features = engineer_features(patient)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        probability = model.predict_proba(
            patient_features
        )[0][1]

    except Exception as e:

        st.error("Prediction failed.")

        st.code(str(e))

        st.stop()


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    prediction = int(
        probability >= MEDICAL_THRESHOLD
    )

    probability_percentage = probability * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("📊 Screening Result")


    # Probability and classification
    result1, result2 = st.columns(2)


    with result1:

        st.metric(
            "Estimated Diabetes Probability",
            f"{probability_percentage:.1f}%"
        )

        st.progress(
            float(min(probability, 1.0))
        )

        st.caption(
            f"Screening threshold: "
            f"{MEDICAL_THRESHOLD:.3f} "
            f"({MEDICAL_THRESHOLD * 100:.1f}%)"
        )


    with result2:

        if prediction == 1:

            st.error(
                "🔴 HIGHER RISK"
            )

            st.write(
                "Further clinical evaluation is recommended."
            )

        else:

            st.success(
                "🟢 LOWER RISK"
            )

            st.write(
                "Continue routine clinical assessment."
            )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("📋 Interpretation")

    if prediction == 1:

        st.warning(
            "The estimated probability is at or above the "
            "screening threshold. This result should be "
            "followed by appropriate clinical evaluation."
        )

    else:

        st.info(
            "The estimated probability is below the screening "
            "threshold. A lower predicted risk does not "
            "guarantee that diabetes is absent."
        )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.subheader("👤 Patient Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Age",
            f"{age} years"
        )

    with c2:
        st.metric(
            "Glucose",
            f"{glucose:.0f} mg/dL"
        )

    with c3:
        st.metric(
            "BMI",
            f"{bmi:.1f}"
        )

    with c4:
        st.metric(
            "Pregnancies",
            f"{pregnancies}"
        )


    # ========================================================
    # FULL MEASUREMENTS
    # ========================================================

    with st.expander("🔎 View Patient Measurements"):

        display_df = patient.T

        display_df.columns = ["Value"]

        st.dataframe(
            display_df,
            use_container_width=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("🧠 About SugarSense")

info1, info2, info3 = st.columns(3)


with info1:

    st.subheader("🤖 Machine Learning")

    st.write(
        "The final screening model is a Support Vector "
        "Machine (SVM) classifier."
    )


with info2:

    st.subheader("🎯 Screening Approach")

    st.write(
        "The screening threshold is optimized to prioritize "
        "identification of potential positive cases."
    )


with info3:

    st.subheader("📊 Prediction")

    st.write(
        "The model generates a probability which is compared "
        "with the selected screening threshold."
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("📈 Model Performance")

st.write(
    "The final model was selected after comparing multiple "
    "machine-learning classification algorithms using "
    "stratified cross-validation."
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Cross-Validation ROC-AUC",
        "0.846"
    )

with metric2:
    st.metric(
        "Test ROC-AUC",
        "0.820"
    )

with metric3:
    st.metric(
        "Test Recall",
        "87.0%"
    )

with metric4:
    st.metric(
        "Test F1 Score",
        "68.6%"
    )

st.info(
    """
    **Why recall matters:** SugarSense is designed as a screening
    tool. Missing a potential diabetes case (false negative) can
    be more concerning in a screening setting, so the decision
    threshold was lowered from the default 0.50 to 0.240.

    This increases sensitivity to potential positive cases while
    accepting that some additional false positives may occur.
    """
)

# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header("⚙️ How It Works")

h1, h2, h3, h4 = st.columns(4)

with h1:

    st.write("**1️⃣ Enter Data**")

    st.caption(
        "Enter the patient's clinical measurements."
    )


with h2:

    st.write("**2️⃣ Process Data**")

    st.caption(
        "The same feature engineering used during training "
        "is applied."
    )


with h3:

    st.write("**3️⃣ Predict**")

    st.caption(
        "The trained SVM estimates diabetes probability."
    )


with h4:

    st.write("**4️⃣ Screen**")

    st.caption(
        "The probability is compared with the screening threshold."
    )


# ============================================================
# LIMITATIONS
# ============================================================

st.divider()

st.header("⚠️ Important Limitations")

st.warning(
    """
    This application is intended for educational and screening
    purposes only. It is not a diagnostic tool and should not
    replace professional medical advice, examination or testing.

    The model was developed using the Pima Indians Diabetes
    Database, representing a specific population. Therefore,
    its performance may not generalize equally to every
    population.

    A healthcare professional should interpret any screening
    result together with symptoms, medical history and
    appropriate clinical testing.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 SugarSense | Early Diabetes Risk Screening | "
    "Machine Learning Project | Educational Use Only"
)