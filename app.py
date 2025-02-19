import os
import pickle
import re
import docx
import pdfplumber
from flask import Flask, render_template, request, redirect, session
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Define a set of common skills (in lowercase)
common_skills = {
    skill.lower()
    for skill in {
        "Python", "Java", "C++", "C", "JavaScript", "HTML", "CSS",
        "TypeScript", "Swift", "Kotlin", "Go", "Ruby", "PHP", "R", "MATLAB",
        "Perl", "Rust", "Dart", "Scala", "Shell Scripting", "React", "Angular",
        "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js",
        "Laravel", "Bootstrap", "TensorFlow", "PyTorch", "Keras",
        "Scikit-learn", "NLTK", "Pandas", "NumPy", "SQL", "MySQL",
        "PostgreSQL", "MongoDB", "Firebase", "Cassandra", "Oracle", "Redis",
        "MariaDB", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "CI/CD", "Jenkins", "Git", "GitHub", "Cybersecurity",
        "Penetration Testing", "Ubuntu", "Ethical Hacking", "Firewalls",
        "Cryptography", "IDS", "Network Security", "Machine Learning",
        "Deep Learning", "Numpy", "Pandas", "Matplotlib", "Computer Vision",
        "NLP", "Big Data", "Hadoop", "Spark", "Data Analytics", "Power BI",
        "Tableau", "Data Visualization", "Reinforcement Learning",
        "Advanced DSA", "DSA", "Data Structures and Algorithm", "DevOps", "ML",
        "DL", "Image Processing", "JIRA", "Postman", "Excel", "Leadership",
        "Problem-Solving", "Communication", "Time Management", "Adaptability",
        "Teamwork", "Presentation Skills", "Critical Thinking",
        "Decision Making", "Public Speaking", "Project Management"
    }
}

abbreviation_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "js": "javascript",
    "html": "hypertext markup language",
    "css": "cascading style sheets",
    "sql": "structured query language",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "dsa": "data structure algorithm"
}


def get_db_connection(db_name):
    """Establishes and returns a connection to the specified MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database=db_name,
        # auth_plugin="mysql_native_password"
    )


# ---------------- Resume Processing Functions ----------------


def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"[ERROR] PDF Processing Failed: {e}")
            return None
    elif file.filename.endswith(".docx"):
        try:
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"[ERROR] DOCX Processing Failed: {e}")
            return None
    else:
        print("[ERROR] Unsupported file format!")
        return None

    return text.strip() if text.strip() else None


def extract_sections(text):
    sections = {
        "summary": None,
        "education": None,
        "work_experience": None,
        "projects": None,
        "skills": None,
        "certifications": None,
        "publications": None,
        "competencies": None,
    }

    section_patterns = {
        "summary": r"(summary|profile|about me)[:\n]",
        "education": r"(education|academic background)[:\n]",
        "work_experience":
        r"(work experience|employment history|professional experience)[:\n]",
        "projects": r"(projects|personal projects|academic projects)[:\n]",
        "skills": r"(skills|technical skills|programming languages)[:\n]",
        "certifications": r"(certifications|courses|training)[:\n]",
        "publications": r"(publications|research papers)[:\n]",
        "competencies": r"(competencies|key competencies|expertise)[:\n]",
    }

    for section, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            next_match = min([
                m.start() for m in [
                    re.search(p, text[start_idx:], re.IGNORECASE)
                    for p in section_patterns.values()
                ] if m
            ],
                             default=len(text))
            sections[section] = text[start_idx:start_idx + next_match].strip()

    return sections


def extract_skills(text):
    extracted_skills = set()
    doc = nlp(text)

    for token in doc:
        word = token.text.lower()  # Convert token to lowercase
        if word in common_skills:
            extracted_skills.add(word)

    return list(extracted_skills)


def extract_name(text):
    lines = text.split('\n')
    if lines:
        return lines[0].strip()  # Assume the first line contains the name
    else:
        return None


def load_model_and_vectorizer():
    model_path = "model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, "rb") as model_file:
            rf = pickle.load(model_file)
        with open(vectorizer_path, "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        print("[INFO] Model and Vectorizer loaded successfully.")
        return rf, tfidf
    else:
        print("[ERROR] Model or Vectorizer missing!")
        return None, None


def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    if rf is None or tfidf is None:
        # Return 5 values consistently
        return "[ERROR] ML model is missing!", None, None, None, None

    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None, None, None

    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    extract_section = extract_sections(text)

    try:
        text_vectorized = tfidf.transform([text])
        print("Vectorized Input Shape:", text_vectorized.shape)
        predicted_job = rf.predict(text_vectorized)[0]
        print("Predicted Job:", predicted_job)

        return None, predicted_job, extracted_skills, extract_section, user_name
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, extract_section, user_name


def normalize_skill(skill):
    """Normalize a skill by converting abbreviations to full forms."""
    skill_lower = skill.lower()  # Handle case sensitivity
    return abbreviation_map.get(skill_lower, skill_lower)


def compare_skills(predicted_job, extracted_skills, user_name):
    try:
        # Connect to skills database
        conn_skills = get_db_connection("skills_db")
        cursor_skills = conn_skills.cursor(dictionary=True)
        cursor_skills.execute(
            "SELECT skills FROM JobRolesSkills WHERE job_role = %s",
            (predicted_job, ))
        job_data = cursor_skills.fetchone()
        cursor_skills.close()
        conn_skills.close()

        if not job_data:
            print("[ERROR] Job role not found in database.")
            return []

        # Normalize required skills from the database
        required_skills = set(
            normalize_skill(skill) for skill in job_data["skills"].split(", "))

        # Normalize extracted skills
        normalized_extracted_skills = set(
            normalize_skill(skill) for skill in extracted_skills)

        missing_skills = required_skills - normalized_extracted_skills

        if missing_skills:
            # Connect to recommended skills database
            conn_mismatch = get_db_connection("recommended_skills_db")
            cursor_mismatch = conn_mismatch.cursor()
            cursor_mismatch.execute(
                "INSERT INTO recommendskills (name, job_role, missing_skills) VALUES (%s, %s, %s)",
                (user_name, predicted_job, ", ".join(missing_skills)))
            conn_mismatch.commit()
            cursor_mismatch.close()
            conn_mismatch.close()

        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison failed: {e}")
        return []


# ---------------- Flask Application ----------------

app = Flask(__name__, template_folder="templates")
app.secret_key = 'firdous'


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None
    extracted_skills = []
    missing_skills = []
    user_name = ""

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                # Unpack all 5 returned values from process_resume
                error_message, predicted_job, extracted_skills, extract_section, user_name = process_resume(
                    file)
                if not error_message:
                    missing_skills = compare_skills(predicted_job,
                                                    extracted_skills,
                                                    user_name)
                    try:
                        conn_resume = get_db_connection("resume_db")
                        cursor_resume = conn_resume.cursor()
                        skills_str = ", ".join(extracted_skills)
                        cursor_resume.execute(
                            "INSERT INTO resumes (name, skills) VALUES (%s, %s)",
                            (user_name, skills_str))
                        conn_resume.commit()
                        cursor_resume.close()
                        conn_resume.close()
                    except Exception as db_error:
                        error_message = f"[ERROR] Database error: {db_error}"

    return render_template("index.html",
                           predicted_job=predicted_job or "",
                           error_message=error_message or "",
                           extracted_skills=extracted_skills,
                           missing_skills=missing_skills,
                           recommended_skills=missing_skills)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not (username and email and password):
            message = "All fields are required!"
            return render_template('signup.html', message=message)

        hashed_password = generate_password_hash(password)

        try:
            # Connect to the users database
            conn = get_db_connection("resume_db")
            cursor = conn.cursor()

            check_query = "SELECT * FROM users_signup_details WHERE email = %s"
            cursor.execute(check_query, (email, ))
            existing_user = cursor.fetchone()

            if existing_user:
                message = "User already exists! Try logging in."
            else:
                insert_query = "INSERT INTO users_signup_details (username, email, password) VALUES (%s, %s, %s)"
                cursor.execute(insert_query,
                               (username, email, hashed_password))
                conn.commit()
                message = "Signup Successful! Now you can login."

            cursor.close()
            conn.close()
        except Exception as e:
            message = f"Error: {e}"

    return render_template('signup.html', message=message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            # Connect to the users database
            conn = get_db_connection("resume_db")
            cursor = conn.cursor()

            query = "SELECT id, username, password FROM users_signup_details WHERE email = %s"
            cursor.execute(query, (email, ))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                user_id, username, stored_password = user
                if check_password_hash(stored_password, password):
                    session['user_id'] = user_id
                    session['username'] = username
                    return redirect('/dashboard')
                else:
                    message = "Invalid Credentials!"
            else:
                message = "Invalid Credentials!"
        except Exception as e:
            message = f"Error: {e}"

    return render_template('login.html', message=message)


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return f"Welcome {session['username']} to your dashboard!"
    return redirect('/login')


if __name__ == "__main__":
    app.run(debug=True)
