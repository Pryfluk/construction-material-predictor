import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model/concrete_estimation_model.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Construction Material Estimator",
    page_icon="🏗️",
    layout="centered"
)

# -----------------------------
# Title & description
# -----------------------------
st.title("🏗️ Construction Material Estimator")
st.write(
    """
    Web application สำหรับประมาณ **ปริมาณคอนกรีตที่ใช้ในการก่อสร้างอาคาร 1 ชั้น**
    โดยใช้ Multiple Linear Regression
    """
)

# -----------------------------
# User inputs
# -----------------------------
st.header("🔢 กรอกข้อมูลโครงสร้าง (ต่อ 1 ชั้น)")

floor_area = st.number_input(
    "พื้นที่ชั้น (ตารางเมตร)",
    min_value=20.0,
    max_value=500.0,
    value=100.0
)

floor_height = st.number_input(
    "ความสูงชั้น (เมตร)",
    min_value=2.5,
    max_value=5.0,
    value=3.0
)

column_count = st.number_input(
    "จำนวนเสา",
    min_value=4,
    max_value=40,
    value=10,
    step=1
)

beam_count = st.number_input(
    "จำนวนคาน",
    min_value=4,
    max_value=50,
    value=12,
    step=1
)

slab_thickness = st.number_input(
    "ความหนาพื้น (เมตร)",
    min_value=0.10,
    max_value=0.30,
    value=0.15
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Concrete Volume"):
    input_data = np.array([
        [
            floor_area,
            floor_height,
            column_count,
            beam_count,
            slab_thickness
        ]
    ])

    prediction = model.predict(input_data)[0]

    st.success(
        f"🧱 ปริมาณคอนกรีตที่คาดว่าจะใช้ ≈ **{prediction:.2f} ลูกบาศก์เมตร**"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Student Project | Multiple Linear Regression | Construction Estimation")
