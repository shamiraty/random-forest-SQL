import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Dataset definition
data = [
    {"program": "business administration", "subjects": ["history", "geography", "maths", "civics", "bookkeeping", "commerce", "economics", "management", "leadership","kiswahili","english"]},
    {"program": "information technology", "subjects": ["maths", "physics", "biology", "electronics", "geography", "programming", "web development", "networking","basic mathematics"]},
    {"program": "medicine", "subjects": ["biology", "chemistry", "physics", "maths", "general studies", "anatomy", "physiology", "biostatistics", "pharmacology"]},
    {"program": "engineering", "subjects": ["maths", "physics", "chemistry", "technical drawing", "electronics", "mechanics", "material science", "design"]},
    {"program": "law", "subjects": ["history", "civics", "english literature", "general studies", "geography", "constitutional law", "criminal law", "ethics"]},
    {"program": "computer science", "subjects": ["maths", "physics", "programming", "electronics", "geography", "data structures", "ai", "machine learning", "algorithms"]},
    {"program": "economics", "subjects": ["maths", "economics", "civics", "history", "geography", "statistics", "microeconomics", "macroeconomics"]},
    {"program": "agriculture", "subjects": ["biology", "chemistry", "agriculture", "geography", "physics", "soil science", "horticulture", "crop production"]},
    {"program": "environmental science", "subjects": ["biology", "geography", "environmental studies", "chemistry", "physics", "ecology", "climate change", "sustainability"]},
    {"program": "biotechnology", "subjects": ["biology", "chemistry", "genetics", "microbiology", "molecular biology", "bioinformatics", "physics", "maths"]},
    {"program": "cyber security", "subjects": ["programming", "networking", "ethical hacking", "maths", "physics", "cryptography", "information security", "ai"]},
    {"program": "digital forensics", "subjects": ["networking", "law", "maths", "programming", "investigation techniques", "cyber crime", "data recovery", "digital systems"]},
    {"program": "education", "subjects": ["psychology", "sociology", "teaching methods", "general studies", "english", "geography", "curriculum development", "research methods"]},
    {"program": "tourism and hospitality", "subjects": ["geography", "civics", "history", "management", "marketing", "event management", "cultural studies", "economics"]},
    {"program": "renewable energy", "subjects": ["physics", "maths", "environmental studies", "chemistry", "renewable resources", "energy systems", "design", "economics"]},
    {"program": "multimedia technology", "subjects": ["graphics design", "video editing", "programming", "physics", "maths", "web design", "animation", "human-computer interaction"]},
    {"program": "art in economics and statistics", "subjects": ["economics", "statistics", "mathematics", "civics", "geography"]},
    {"program": "art in environmental economics and policy", "subjects": ["environmental studies", "economics", "policy making", "geography", "civics"]},
    {"program": "arts in economics", "subjects": ["economics", "mathematics", "history", "geography", "philosophy"]},
    {"program": "arts in economics and sociology", "subjects": ["economics", "sociology", "psychology", "mathematics", "history","social study"]},
    {"program": "business administration", "subjects": ["accounting", "marketing", "finance", "human resources", "entrepreneurship"]},
    {"program": "commerce in accounting", "subjects": ["accounting", "auditing", "taxation", "corporate law", "finance"]},
    {"program": "commerce in entrepreneurship", "subjects": ["entrepreneurship", "marketing", "business strategy", "finance", "accounting"]},
    {"program": "commerce in finance", "subjects": ["finance", "economics", "investments", "banking", "taxation"]},
    {"program": "commerce in human resource management", "subjects": ["human resources", "organizational behavior", "labor laws", "psychology", "communication skills"]},
    {"program": "commerce in international business", "subjects": ["international trade", "global economics", "foreign policy", "marketing", "supply chain"]},
    {"program": "commerce in marketing", "subjects": ["marketing", "digital marketing", "brand management", "consumer behavior", "sales","economics"]},
    {"program": "education in administration and management", "subjects": ["educational management", "leadership", "policy planning", "human resources", "statistics"]},
    {"program": "education in adult education and community", "subjects": ["adult learning", "community development", "policy implementation", "sociology", "psychology"]},
    {"program": "education in arts", "subjects": ["fine arts", "history", "literature", "philosophy", "psychology"]},
    {"program": "education in science", "subjects": ["biology", "chemistry", "physics", "mathematics", "environmental studies"]},
    {"program": "science in computer engineering", "subjects": ["programming", "electronics", "networking", "system design", "artificial intelligence"]},
    {"program": "science in cyber security", "subjects": ["cyber security", "digital forensics", "ethical hacking", "data privacy", "information security"]},
    {"program": "science in renewable energy engineering", "subjects": ["renewable energy", "environmental science", "physics", "chemistry", "engineering mathematics"]},
    {"program": "science in applied geology", "subjects": ["geology", "geophysics", "environmental science", "mining", "geochemistry"]},
    {"program": "science in statistics", "subjects": ["probability", "statistical analysis", "mathematics", "data science", "economics"]},
    {"program": "science in software engineering", "subjects": ["software development", "algorithms", "database systems", "operating systems", "cloud computing"]},
    {"program": "science in telecommunications engineering", "subjects": ["telecommunications", "signal processing", "networking", "electronics", "wireless systems"]},
    {"program": "philosophy in economics", "subjects": ["advanced economics", "economic theory", "research methodology", "quantitative analysis", "policy development"]},
    {"program": "philosophy in business administration", "subjects": ["business strategy", "leadership", "corporate governance", "financial management", "operations management","politics"]},
]



# Prepare DataFrame
df = pd.DataFrame(data)
mlb = MultiLabelBinarizer()
subject_matrix = mlb.fit_transform(df['subjects'])
subject_columns = mlb.classes_

# Add subjects as binary features
subject_df = pd.DataFrame(subject_matrix, columns=subject_columns)
df = pd.concat([df, subject_df], axis=1)

# Split Data
X = df[subject_columns]
y = df["program"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
# Streamlit App
st.title("Random Forest Classifier")
st.subheader("The Classification and Regression Trees (CART) Algorithm")  # Subtitle
st.write("Enter the subjects you have completed in high school, and we will recommend the top undergraduate programs for you.")



with st.form(key="subject_form"):
    subjects_input = st.text_input("Enter subjects (comma separated)", "")
    submit_button = st.form_submit_button(label="Get Recommendations")

def validate_subjects(input_subjects):
    """Check if all entered subjects are valid."""
    valid_subjects = set(subject_columns)
    return all(subject in valid_subjects for subject in input_subjects)

if submit_button and subjects_input:
    student_subjects = [subject.strip() for subject in subjects_input.split(",")]

    # Ensure that at least 3 subjects are entered
    if len(student_subjects) < 3:
        st.error("Please enter at least 3 subjects, separated by commas. Note: The input is case-sensitive.")
    else:
        # Validate entered subjects
        if not validate_subjects(student_subjects):
            st.error("Some entered subjects are not recognized. Please check your inputs.")
        else:
            # Prepare input data for prediction
            input_data = [1 if subject in student_subjects else 0 for subject in subject_columns]
            proba = model.predict_proba([input_data])[0]
            programs_with_proba = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)

            st.write("Top 5 Recommended Programs:")
            for program, probability in programs_with_proba[:5]:
             st.warning(f"{program}:  [   {probability * 100:.1f}%  ]")
