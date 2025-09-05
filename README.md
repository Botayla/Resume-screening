📄 **Resume Screening with BERT**
🔍 **Overview**

This project is a Resume Screening System that uses NLP (Natural Language Processing) and BERT-based models to automatically classify resumes into different professional fields and extract relevant skills.

The system aims to save recruiters’ time by automatically identifying the most relevant domain for each candidate and highlighting their key skills.

⚙️ **Features**

📑 Resume Parsing: Extracts raw text from uploaded PDF resumes.

🤖 Resume Classification: Uses a fine-tuned DistilBERT model to classify resumes into fields (e.g., IT, HR, Finance, Data Science, etc.).

🛠 Skill Extraction: Identifies technical and domain-specific skills mentioned in resumes using NLP (SpaCy + RapidFuzz).

🌐 Streamlit Web App: Simple user interface to upload resumes and display predictions.

🏗️ **Project Workflow**

PDF to Text → Extracts text from uploaded resumes using PyMuPDF.

Load Model → Loads the fine-tuned DistilBERT model and label_encoder.pkl for field classification.

Classification → Predicts the professional field of the resume.

Skill Extraction → Uses a domain-specific dictionary + fuzzy matching to identify candidate skills.

UI → Streamlit displays the predicted field and extracted skills.

📦 **Tech Stack**

Python

Transformers (Hugging Face) – DistilBERT for classification

PyMuPDF (fitz) – PDF text extraction

spaCy – NLP pipeline

RapidFuzz – Fuzzy skill matching

Joblib – For label encoder persistence

Streamlit – Frontend web app

🚀 **How to Run the Project**

Clone the repository or download the code.

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app_st.py


Upload a resume in PDF format and view results.

📊 **Example Output**

Predicted Field: Data Science

Extracted Skills: Python, Machine Learning, SQL, Pandas, TensorFlow

📌 **Future Improvements**

Improve skill extraction using NER-based approaches (SkillNER, spaCy custom NER).

Add support for multi-label classification (resumes can belong to multiple fields).

Deploy as a cloud-based API for integration with recruitment systems.

👩‍💻 ****Author****

Developed by Botayla Amin

Kaggle Notebook: Resume Screening with BERT
