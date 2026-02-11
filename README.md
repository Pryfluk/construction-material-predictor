# 🏗️ Construction Material Estimator (1-Floor Building)

โปรเจกต์นี้เป็นส่วนหนึ่งของงานเดี่ยวในรายวิชา  
มีวัตถุประสงค์เพื่อพัฒนา **Web Application** สำหรับประมาณ  
**ปริมาณคอนกรีตที่ใช้ในการก่อสร้างอาคาร 1 ชั้น**  
โดยใช้เทคนิค **Multiple Linear Regression**

---

## 📌 Objective
- เพื่อศึกษาการประยุกต์ใช้ Multiple Linear Regression กับงานด้านการก่อสร้าง
- เพื่อพัฒนา Machine Learning Pipeline ตั้งแต่การเตรียมข้อมูล การฝึกโมเดล การประเมินผล และการบันทึกโมเดล
- เพื่อพัฒนา Web Application ที่ผู้ใช้สามารถกรอกข้อมูลโครงสร้าง และรับค่าประมาณปริมาณคอนกรีตได้

---

## 📊 Dataset
- จำนวนข้อมูลทั้งหมด: **500 samples**
- แหล่งที่มาของข้อมูล:  
  ข้อมูลถูกสร้างขึ้นจากหลักการประมาณปริมาณวัสดุก่อสร้างเบื้องต้น (Simulated Data)
- ประเภทข้อมูล: Numerical data

### 🔹 Features
| Feature | Description |
|------|------------|
| floor_area | พื้นที่ชั้น (ตารางเมตร) |
| floor_height | ความสูงของชั้น (เมตร) |
| column_count | จำนวนเสา |
| beam_count | จำนวนคาน |
| slab_thickness | ความหนาพื้น (เมตร) |

### 🔹 Target
| Target | Description |
|------|------------|
| concrete_volume | ปริมาณคอนกรีตที่ใช้ (ลูกบาศก์เมตร) |

---

## 🧠 Machine Learning Model
- Model: **Multiple Linear Regression**
- Train/Test Split: **80% / 20%**
- Evaluation Metrics:
  - Mean Squared Error (MSE)
  - R-squared (R² Score)

---

## 🗂️ Project Structure

construction-material-estimator/
│
├── data/
│ └── construction_training_data.csv
│
├── model/
│ └── concrete_estimation_model.pkl
│ └── model_evaluation.txt
│
├── .gitignore
├── app.py
├── model.py
├── README.md
└── requirements.txt


---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

python train_model.py

python -m streamlit run app.py
