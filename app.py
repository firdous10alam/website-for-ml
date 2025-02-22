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

# Define common skills
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


# Database Connection
def get_db_connection(db_name):
    """Establishes and returns a connection to the specified MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="PHW#84#jeorr",
        database=db_name,
        # auth_plugin="mysql_native_password"
    )


# Resume Processing Functions
def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip() if text.strip() else None


def extract_skills(text):
    extracted_skills = set()
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in common_skills:
            extracted_skills.add(token.text.lower())
    return list(extracted_skills)


def extract_name(text):
    lines = text.split('\n')
    return lines[0].strip() if lines else None


def load_model_and_vectorizer():
    try:
        with open("model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        return rf, tfidf
    except Exception as e:
        print(f"[ERROR] Failed to load model/vectorizer: {e}")
        return None, None


def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    if not rf or not tfidf:
        return "[ERROR] ML model is missing!", None, None, None

    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None, None

    user_name = extract_name(text)
    extracted_skills = extract_skills(text)

    try:
        text_vectorized = tfidf.transform([text])
        predicted_job = rf.predict(text_vectorized)[0]
        return None, predicted_job, extracted_skills, user_name
    except Exception as e:
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, user_name


def compare_skills(predicted_job, extracted_skills, user_name):
    try:
        conn = get_db_connection('resume_screening_db')
        cursor = conn.cursor(dictionary=True)

        # Fetch required skills for the predicted job
        cursor.execute("SELECT skills FROM jobrolesskills WHERE job_role = %s",
                       (predicted_job, ))
        job_data = cursor.fetchone()

        if not job_data:
            return []

        required_skills = set(job_data["skills"].lower().split(", "))
        extracted_skills_set = set(skill.lower() for skill in extracted_skills)
        missing_skills = required_skills - extracted_skills_set

        if missing_skills:
            cursor.execute(
                "INSERT INTO recommendskills (name, job_role, missing_skills) VALUES (%s, %s, %s)",
                (user_name, predicted_job, ", ".join(missing_skills)))
            conn.commit()

        cursor.close()
        conn.close()
        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison failed: {e}")
        return []


# Flask Application
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
                error_message, predicted_job, extracted_skills, user_name = process_resume(
                    file)
                if not error_message:
                    missing_skills = compare_skills(predicted_job,
                                                    extracted_skills,
                                                    user_name)
                    try:
                        conn = get_db_connection('resume_screening_db')
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO resumes (name, skills) VALUES (%s, %s)",
                            (user_name, ", ".join(extracted_skills)))
                        conn.commit()
                        cursor.close()
                        conn.close()
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
            conn = get_db_connection("signup_db")
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
            conn = get_db_connection("signup_db")
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
