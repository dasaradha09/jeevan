import streamlit as st
import numpy as np
import joblib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping dictionaries
marital_status_map = {
    "Single": 1, "Married": 2, "Widower": 3, "Divorced": 4,
    "Facto union": 5, "Legally separated": 6
}

application_mode_map = {
    "1st phase—general contingent": 1, "Ordinance No. 612/93": 2,
    "1st phase—special contingent (Azores Island)": 3, "Holders of other higher courses": 4,
    "Ordinance No. 854-B/99": 5, "International student (bachelor)": 6,
    "1st phase—special contingent (Madeira Island)": 7, "2nd phase—general contingent": 8,
    "3rd phase—general contingent": 9, "Ordinance No. 533-A/99 (Different Plan)": 10,
    "Ordinance No. 533-A/99 (Other Institution)": 11, "Over 23 years old": 12,
    "Transfer": 13, "Change in course": 14, "Technological specialization diploma holders": 15,
    "Change in institution/course": 16, "Short cycle diploma holders": 17,
    "Change in institution/course (International)": 18
}

course_map = {
    "Biofuel Production Technologies": 1, "Animation and Multimedia Design": 2,
    "Social Service (evening attendance)": 3, "Agronomy": 4, "Communication Design": 5,
    "Veterinary Nursing": 6, "Informatics Engineering": 7, "Equiniculture": 8,
    "Management": 9, "Social Service": 10, "Tourism": 11, "Nursing": 12,
    "Oral Hygiene": 13, "Advertising and Marketing Management": 14, "Journalism and Communication": 15,
    "Basic Education": 16, "Management (evening attendance)": 17
}

previous_qualification_map = {
    "Secondary education": 1, "Higher education—bachelor’s degree": 2, "Higher education—degree": 3,
    "Higher education—master’s degree": 4, "Higher education—doctorate": 5,
    "Frequency of higher education": 6, "12th year of schooling—not completed": 7,
    "11th year of schooling—not completed": 8, "Other—11th year of schooling": 9,
    "10th year of schooling": 10, "10th year—not completed": 11,
    "Basic education 3rd cycle": 12, "Basic education 2nd cycle": 13,
    "Technological specialization course": 14, "Higher education—degree (1st cycle)": 15,
    "Professional higher technical course": 16, "Higher education—master’s degree (2nd cycle)": 17
}

# Streamlit UI
st.title("Student Status Prediction (with PCA fields only)")

# Input fields
marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
application_mode = st.selectbox("Application Mode", list(application_mode_map.keys()))
application_order = st.number_input("Application Order", min_value=1, step=1)
course = st.selectbox("Course", list(course_map.keys()))
daytime_evening = st.selectbox("Attendance Time", ["Daytime", "Evening"])
previous_qualification = st.selectbox("Previous Qualification", list(previous_qualification_map.keys()))
mother_qualification = st.selectbox("Mother's Qualification (code)", list(range(1, 35)))
father_occupation = st.selectbox("Father's Occupation (code)", list(range(1, 47)))
displaced = st.selectbox("Displaced", ["Yes", "No"])
special_needs = st.selectbox("Educational Special Needs", ["Yes", "No"])
debtor = st.selectbox("Debtor", ["Yes", "No"])
tuition_up_to_date = st.selectbox("Tuition Fees Up To Date", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
scholarship_holder = st.selectbox("Scholarship Holder", ["Yes", "No"])
age_at_enrollment = st.slider("Age at Enrollment", 15, 60, 18)
international = st.selectbox("International Student", ["Yes", "No"])
units_grade = st.number_input("Curricular Units Grade (PCA)", min_value=0.0, step=0.01)

# Email function
def send_email(subject, body, to_email):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "masterpiece2124@gmail.com"
    password = "fljl urum putc jkau"

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        st.success(f"Email sent successfully to {to_email}")
    except Exception as e:
        st.error(f"Error sending email: {e}")
    finally:
        server.quit()

# Prediction logic
if st.button("Predict Student Status"):
    features = np.array([[ 
        marital_status_map[marital_status],
        application_mode_map[application_mode],
        application_order,
        course_map[course],
        1 if daytime_evening == "Daytime" else 0,
        previous_qualification_map[previous_qualification],
        mother_qualification,
        father_occupation,
        1 if displaced == "Yes" else 0,
        1 if special_needs == "Yes" else 0,
        1 if debtor == "Yes" else 0,
        1 if tuition_up_to_date == "Yes" else 0,
        1 if gender == "Male" else 0,
        1 if scholarship_holder == "Yes" else 0,
        age_at_enrollment,
        1 if international == "Yes" else 0,
        units_grade
    ]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    target_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    predicted_status = target_map.get(prediction, "Unknown")
    st.success(f"Predicted Student Status: {predicted_status}")

    st.session_state["last_prediction"] = prediction
    st.session_state["show_email_input"] = prediction == 0

# Email input and sending logic (persisted)
if st.session_state.get("show_email_input", False):
    parent_email = st.text_input("Enter Parent's Email", placeholder="e.g. abc@gmail.com")

    if st.button("Send Email Notification"):
        if parent_email:
            subject = "Alert: Student Predicted to Drop Out"
            body = (
                "Dear Parent,\n\n"
                "This is to inform you that your child is predicted to be at risk of dropping out. "
                "Please contact the academic advisor for more information and support options.\n\n"
                "Regards,\nUniversity Analytics Team"
            )
            st.info(f"Sending Email to {parent_email}...")
            send_email(subject, body, parent_email)
        else:
            st.warning("Please enter a valid parent email before sending.")