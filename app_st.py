import streamlit as st
import fitz  # PyMuPDF
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz, process



# ----------------------------
# 1. PDF to Text
# ----------------------------
def pdf_to_text(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ----------------------------
# 2. Load Model & Tokenizer
# ----------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(r"H:\Faculty_of_AI\level3\Project NLP")
    model = DistilBertForSequenceClassification.from_pretrained(
        r"H:\Faculty_of_AI\level3\Project NLP", local_files_only=True
    )
    label_encoder = joblib.load(r"H:\Faculty_of_AI\level3\Project NLP\label_encoder.pkl")
    return tokenizer, model, label_encoder

# ----------------------------
# 3. Classify Resume
# ----------------------------
def classify_resume(text, tokenizer, model, label_encoder):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()
    return label_encoder.inverse_transform(preds)[0]

# ----------------------------
# 4. Extract Skills (SkillNer)
# ----------------------------
# 1.
nlp = spacy.load("en_core_web_sm")

# 2. skills dictionary
skills_dict = {
    "INFORMATION-TECHNOLOGY": [
        "python", "java", "c++", "sql", "machine learning", "deep learning", "cloud", "linux",
        "docker", "kubernetes", "aws", "azure", "git", "javascript", "html", "css", "react", "node.js","excel",
        "power bi"
    ],
    "AGRICULTURE": [
        "crop management", "soil science", "irrigation", "fertilizers", "pesticides",
        "agronomy", "farm equipment", "organic farming", "supply chain"
    ],
    "DESIGNER": [
        "photoshop", "illustrator", "figma", "adobe xd", "ui design", "ux design", "sketch",
        "wireframing", "prototyping", "indesign", "after effects"
    ],
    "FINANCE": [
        "accounting", "financial analysis", "budgeting", "auditing", "taxation", "excel",
        "investment", "risk management", "forecasting", "financial modeling"
    ],
    "SALES": [
        "lead generation", "cold calling", "negotiation", "b2b sales", "crm", "business development",
        "market research", "account management", "salesforce", "pipeline management"
    ],
    "HEALTHCARE": [
        "patient care", "clinical research", "public health", "nursing", "epidemiology", "pharmacy",
        "diagnosis", "treatment planning", "medical terminology"
    ],
    "TEACHER": [
        "curriculum design", "lesson planning", "classroom management", "assessment", "tutoring",
        "e-learning", "educational technology", "research", "academic writing"
    ],
    "CONSTRUCTION": [
        "project management", "autocad", "safety management", "civil engineering", "blueprint reading",
        "contract management", "site supervision", "structural analysis"
    ],
    "ENGINEERING": [
        "matlab", "solidworks", "autocad", "ansys", "circuit design", "mechanical design",
        "electrical systems", "embedded systems", "quality assurance"
    ],
    "BUSINESS-DEVELOPMENT": [
        "strategic planning", "partnerships", "market analysis", "growth strategy", "sales strategy",
        "negotiation", "stakeholder management", "competitive analysis"
    ],
    "ACCOUNTANT": [
        "bookkeeping", "tax preparation", "accounts payable", "accounts receivable", "general ledger",
        "auditing", "financial reporting", "quickbooks", "sap", "excel"
    ],
    "BANKING": [
        "financial services", "credit analysis", "risk assessment", "loan processing", "compliance",
        "anti-money laundering", "customer service", "retail banking", "investment banking"
    ],
    "DATA-SCIENCE": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "data visualization",
        "matplotlib", "seaborn", "statistics", "nlp", "big data"
    ],
    "CONSULTANT": [
        "business analysis", "management consulting", "problem solving", "presentation skills",
        "strategic planning", "financial modeling", "stakeholder engagement"
    ],
    "ARTS": [
        "drawing", "painting", "digital art", "photography", "illustration", "creative writing",
        "film editing", "3d modeling", "animation"
    ],
    "AVIATION": [
        "aircraft maintenance", "flight operations", "navigation", "aerospace engineering",
        "safety management", "aviation regulations", "air traffic control"
    ],
    "AUTOMOBILE": [
        "automotive engineering", "vehicle maintenance", "mechanical systems", "quality testing",
        "manufacturing", "solidworks", "cad", "engine diagnostics"
    ],
    "CHEMICAL": [
        "process engineering", "lab testing", "polymer science", "reaction engineering",
        "chemical safety", "spectroscopy", "analytical chemistry"
    ],
    "ELECTRICAL-ENGINEERING": [
        "circuit design", "embedded systems", "power systems", "control systems", "pcb design",
        "arduino", "fpga", "signal processing"
    ],
    "OPERATIONS": [
        "supply chain", "inventory management", "logistics", "lean manufacturing",
        "six sigma", "process improvement", "operations strategy"
    ],
    "PUBLIC-RELATIONS": [
        "media relations", "event management", "press releases", "crisis communication",
        "branding", "digital marketing", "storytelling", "content creation"
    ],
    "SAP": [
        "sap fico", "sap mm", "sap sd", "sap abap", "sap hana", "erp systems",
        "business process", "configuration"
    ],
    "SUPPLY-CHAIN": [
        "procurement", "inventory management", "logistics", "warehouse management",
        "forecasting", "demand planning", "supplier management"
    ],
    "NETWORKING": [
        "cisco", "network security", "firewalls", "routing", "switching", "vpn",
        "wireless networking", "tcp/ip", "troubleshooting"
    ],
    # فئات الموديل اللي مكنتش موجودة عندك
    "ADVOCATE": [
        "legal research", "litigation", "contracts", "legal writing", "compliance",
        "intellectual property", "court representation"
    ],
    "APPAREL": [
        "fashion design", "textile", "garment production", "sewing", "pattern making",
        "merchandising", "styling"
    ],
    "BPO": [
        "customer service", "call center", "outsourcing", "process management",
        "technical support", "telemarketing"
    ],
    "CHEF": [
        "cooking", "menu planning", "food safety", "culinary arts", "baking",
        "kitchen management", "plating"
    ],
    "DIGITAL-MEDIA": [
        "social media marketing", "content creation", "seo", "sem", "influencer marketing",
        "video production", "digital strategy"
    ],
    "FITNESS": [
        "personal training", "nutrition", "exercise physiology", "strength training",
        "yoga", "fitness assessment", "rehabilitation"
    ],
    "HR": [
        "recruitment", "talent acquisition", "employee relations", "hr policies", "training",
        "payroll", "performance management", "conflict resolution", "onboarding"
    ]
}


# 3. Function to extract skills based on predicted field
def extract_skills(text, predicted_field):
    skills = skills_dict.get(predicted_field, [])
    found_skills = []
    for skill in skills:
        if fuzz.partial_ratio(skill.lower(), text.lower()) > 80:  # threshold
            found_skills.append(skill)
    return found_skills

# ----------------------------
# 5. Streamlit UI
# ----------------------------
st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")

st.title("📄 Resume Screening with DistilBERT + Skill Extraction")

uploaded_pdf = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_pdf:
    text = pdf_to_text(uploaded_pdf)
    text = text.lower()
    tokenizer, model, label_encoder = load_model()
    # skill_extractor = load_skill_extractor()

    # Classification
    prediction = classify_resume(text, tokenizer, model, label_encoder)

    # Skills
    skills_found = extract_skills(text, prediction)

    st.subheader("🎯 Predicted Field")
    st.success(prediction)

    st.subheader("🛠 Extracted Skills")
    st.write(", ".join(skills_found) if skills_found else "No skills detected")
