import sqlite3
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import fuzz

# Connect to SQLite database
DATABASE = "datas.db"

def create_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

# Fetch data from SQLite and normalize
def fetch_data():
    conn = create_connection()
    programme_query = "SELECT * FROM programme"
    subject_query = """
    SELECT subject.id, LOWER(subject.name) AS subject_name, programme.name AS programme_name
    FROM subject
    JOIN programme ON subject.programme_id = programme.id
    """
    programme_df = pd.read_sql_query(programme_query, conn)
    subject_df = pd.read_sql_query(subject_query, conn)
    conn.close()
    return programme_df, subject_df

# Prepare DataFrame for Machine Learning
def prepare_data(subject_df):
    mlb = MultiLabelBinarizer()
    grouped = subject_df.groupby("programme_name")["subject_name"].apply(list).reset_index()
    subject_matrix = mlb.fit_transform(grouped["subject_name"])
    subject_columns = mlb.classes_
    
    subject_df_ml = pd.DataFrame(subject_matrix, columns=subject_columns)
    programme_df_ml = pd.concat([grouped["programme_name"], subject_df_ml], axis=1)
    return programme_df_ml, subject_columns

# Train Random Forest Model
def train_model(data, subject_columns):
    X = data[subject_columns]
    y = data["programme_name"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to validate subjects with fuzzy matching
def validate_subjects_fuzzy(input_subjects, valid_subjects, threshold=70):
    valid_subjects = [subject.lower() for subject in valid_subjects]
    matched_subjects = []
    
    for input_subject in input_subjects:
        matched = False
        for valid_subject in valid_subjects:
            similarity = fuzz.ratio(input_subject.lower(), valid_subject)
            if similarity >= threshold:
                matched_subjects.append(valid_subject)
                matched = True
                break
        if not matched:
            return False, matched_subjects
    return True, matched_subjects

# Streamlit App
st.title("RANDOM FOREST CLASSFIER")
st.subheader("The Classification and Regression Trees (CART) Algorithm")  # Subtitle
st.write("Enter the subjects you have completed in high school, and we will recommend the top undergraduate programs for you.")

# Load and prepare data
programme_df, subject_df = fetch_data()
data, subject_columns = prepare_data(subject_df)
model = train_model(data, subject_columns)

with st.form(key="subject_form"):
    subjects_input = st.text_input("Enter subjects (comma separated)", "")
    submit_button = st.form_submit_button(label="Get Recommendations")

if submit_button and subjects_input:
    student_subjects = [subject.strip().lower() for subject in subjects_input.split(",")]

    if len(student_subjects) < 3:
        st.error("Please enter at least 3 subjects, separated by commas.")
    else:
        valid_subjects = set(subject_columns)
        is_valid, matched_subjects = validate_subjects_fuzzy(student_subjects, valid_subjects)

        if not is_valid:
            st.error(f"Some entered subjects are not recognized or do not match closely enough. Matched subjects: {matched_subjects}")
        else:
            input_data = [1 if subject in matched_subjects else 0 for subject in subject_columns]
            probabilities = model.predict_proba([input_data])[0]
            recommendations = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)

            st.write("Top 5 Recommended Programs:")
            for program, prob in recommendations[:5]:
                st.info(f"{program}:  [   {prob * 100:.1f}%  ]")
