import streamlit as st
import joblib
import PyPDF2

# -----------------------------
# Load the saved model and TF-IDF vectorizer
# -----------------------------
@st.cache_data(show_spinner=False)
def load_model_and_vectorizer():
    try:
        model = joblib.load("resume_classifier.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        return model, tfidf
    except FileNotFoundError:
        st.error("Error: Model or vectorizer file not found.")
        st.stop()

model, tfidf = load_model_and_vectorizer()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 Resume Classifier App")
st.write("Upload your resume (PDF) or paste text below to predict its category.")

# -----------------------------
# File uploader for PDF resumes
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

resume_text = ""

if uploaded_file is not None:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text() + "\n"
        st.success("PDF successfully loaded! You can now predict the category.")
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")

# -----------------------------
# Text area for manual input
# -----------------------------
manual_text = st.text_area("Or paste resume text here", height=200)
if manual_text.strip() != "":
    resume_text = manual_text

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please provide resume text (paste or upload PDF).")
    else:
        try:
            # Transform text into TF-IDF features
            input_features = tfidf.transform([resume_text])
            # Predict category
            prediction = model.predict(input_features)
            st.success(f"Predicted Category: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
