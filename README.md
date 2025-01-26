## RANDOM FOREST CLASSIFIER ML
### THE CLASSIFICATION AND REGRESSION TREES (CART) ALGORITHM
#### HIGH SCHOOL SUBJECT TO PROGRAM RECOMMENDATION

## Live Demo
You can try the **Random Forest Classifier** application for program recommendations using the following live demo link:
[**Live Demo: Random Forest Classifier**](https://random-forest-sql.streamlit.app/)

## Introduction

The **Random Forest Classifier** is a robust and versatile machine learning algorithm that uses an ensemble of decision trees to make predictions. In this application, the model is used to recommend undergraduate programs based on the subjects completed by students in high school. The system leverages data from an SQLite database, employs **fuzzy string matching** to handle variations in subject names, and provides users with a personalized recommendation of programs they might be suited for.

---

## Problem Statement

Students often face challenges when choosing undergraduate programs, as they may not know which programs align with their completed high school subjects. This decision-making process is further complicated by variations in how subjects are named or entered, such as typographical errors or inconsistent naming conventions. This app aims to solve this problem by providing program recommendations based on high school subjects and correcting any discrepancies in the user input.

---

## Main Objective

The main objective of this project is to develop an application that:
1. Recommends top undergraduate programs based on the high school subjects entered by the student.
2. Provides accurate results even if the user makes slight errors or variations in subject names, thanks to fuzzy matching.
3. Uses the **Random Forest Classifier** to generate recommendations based on subject patterns in the dataset.

---

## Specific Objectives

1. To preprocess and prepare data from the SQLite database into a format suitable for machine learning.
2. To train a **Random Forest Classifier** model that can predict the best programs for a student based on their subjects.
3. To validate the entered subjects using fuzzy string matching to ensure that subject names are correctly recognized.
4. To display the top 5 recommended programs along with their prediction probabilities.

---

## Methodology

The application follows these steps:

1. **Data Collection and Preprocessing**:
   - Data is fetched from an SQLite database containing `programme` and `subject` tables. The subject data is normalized to lowercase to maintain uniformity.
   - The `programme` and `subject` data is then transformed into a one-hot encoded format where each subject is represented as a binary vector.

2. **Training the Random Forest Model**:
   - The data is split into training and testing sets. A **Random Forest Classifier** is trained using the subjects as features and the programs as the target.
   - The model is evaluated on the test set, ensuring it can make accurate predictions.

3. **Fuzzy Matching for Subject Validation**:
   - When a user enters their high school subjects, the input is matched against the available subjects in the database using the **fuzzywuzzy** library. This ensures that minor errors in subject names do not affect the prediction.

4. **Program Recommendation**:
   - Once the subjects are validated, the model predicts the likelihood of the student being successful in different programs.
   - The top 5 recommended programs are displayed based on the highest predicted probabilities.

---

## Methodology Details

### 1. **Database Connection and Data Fetching**
   The application connects to the SQLite database (`datas.db`) and fetches the `programme` and `subject` data, which is used to train the machine learning model.

### 2. **Data Preparation**
   - A **MultiLabelBinarizer** is used to convert the subjects into binary features, suitable for training the Random Forest model.
   - The data is split into training and testing sets, and the model is trained to predict which program a student is most likely suited for, given their subjects.

### 3. **Model Training**
   The **Random Forest Classifier** is trained on the prepared data. It learns to associate a set of subjects with a particular program. The model is trained using scikit-learn's `train_test_split` for data splitting and `RandomForestClassifier` for classification.

### 4. **Fuzzy Matching**
   To handle subject variations or typos, **fuzzywuzzy** is employed to match the user’s input with the nearest valid subjects in the database. If the match score is below a certain threshold (e.g., 70%), an error message is displayed.

### 5. **User Interaction via Streamlit**
   - **User Input**: The user enters their completed subjects separated by commas.
   - **Recommendations**: Upon submitting the subjects, the model outputs the top 5 recommended programs along with their prediction probabilities.

---

## Technologies Used

- **Streamlit**: Used for creating an interactive and easy-to-use frontend for the application.
- **Pandas**: For data manipulation and handling the database queries.
- **SQLite**: A lightweight relational database for storing program and subject data.
- **scikit-learn**: For implementing the Random Forest Classifier and preprocessing the data.
- **fuzzywuzzy**: Used for fuzzy string matching to account for variations or typos in subject names.

---

## Expected Outcome

The application provides students with an easy way to enter their high school subjects and receive program recommendations. The top 5 recommended programs will be shown with their respective probabilities, allowing students to make an informed decision about their next academic steps.

---

## Limitations

- The system relies heavily on the data available in the database. If the data is sparse or not representative, the model's recommendations may not be accurate.
- The fuzzy matching technique might not be able to match every possible variation of a subject name, depending on how distinct the subject names are.

---

## Future Plans

- Integrating more subject-related features such as subject difficulty or interest areas could improve the program recommendations.
- The addition of more robust error handling and validation features would ensure a better user experience.
- Potentially expand the model to handle other factors like student preferences or geographical constraints for a more personalized recommendation system.

---

## Conclusion

The **Random Forest Classifier** model combined with **fuzzy string matching** provides an effective solution to recommending programs based on high school subjects. By leveraging machine learning and fuzzy matching, the application offers accurate and personalized program recommendations, helping students make better-informed decisions about their academic future.
