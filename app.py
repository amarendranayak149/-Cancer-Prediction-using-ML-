import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
def load_data():
    return pd.read_csv('cancer_prediction_data (2).csv')

# Data Preprocessing
def preprocess_data(df):
    numeric = ['Age', 'Tumor_Size']
    ordinal = ['Tumor_Grade', 'Symptoms_Severity', 'Alcohol_Consumption', 'Exercise_Frequency']
    nominal = ['Gender', 'Family_History', 'Smoking_History']
    
    numeric_preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    ordinal_preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    nominal_preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocess = ColumnTransformer([
        ('num', numeric_preprocess, numeric),
        ('ord', ordinal_preprocess, ordinal),
        ('nom', nominal_preprocess, nominal)
    ], remainder='passthrough')
    
    X = df.drop('Cancer_Present', axis=1)
    y = df['Cancer_Present']
    return train_test_split(X, y, test_size=0.2, random_state=23), preprocess

# Train Models
def train_model(X_train, y_train, preprocess, model_name):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier()
    }
    model = models.get(model_name, LogisticRegression())  # Default to Logistic Regression
    pipeline = Pipeline([
        ('preprocessor', preprocess),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Custom Styling
st.markdown("""
    <style>
        .stApp {background-color: #121212; color: #FFFFFF;}
        .title {color: #FFD700; text-align: center; font-size: 45px; font-weight: bold; text-shadow: 2px 2px 4px #000000;}
        .prediction {font-size: 30px; font-weight: bold; color: #E74C3C; text-align: center;}
        .safe {font-size: 30px; font-weight: bold; color: #1DB954; text-align: center;}
        .alert-box {
            background-color: #ffcccc;
            border-left: 5px solid #cc0000;
            padding: 15px;
            font-size: 20px;
            color: #cc0000;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# App Title with Enhanced Styling
st.image("innomatics_logo.png", width=350)
st.markdown('<h1 class="title">🎗️ Cancer Prediction 🎗️</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:22px; color:#555;">An intelligent system for early cancer detection using machine learning.</p>', unsafe_allow_html=True)

# Load and Preprocess Data
df = load_data()
(X_train, X_test, y_train, y_test), preprocess = preprocess_data(df)

# Model Selection with Dropdown
model_name = st.selectbox("🧠 Choose a Machine Learning Model", [
    "Decision Tree", "SVM", "Logistic Regression", "KNN"
])

# Train and Evaluate Model
if st.button("🚀 Train & Evaluate Model", key="train_button_unique"):
    with st.spinner("🚀 Training the model... Hang tight! ⏳"):
        model = train_model(X_train, y_train, preprocess, model_name)
        accuracy = model.score(X_test, y_test)
        st.session_state['trained_model'] = model
        st.success(f"✅ Model trained successfully with an accuracy of **{accuracy:.2f}**! 🎯")

# Prediction Section with Custom Styling
st.markdown('<h2 style="color: #D32F2F; text-align: center; font-size: 30px;">🔍 Make a Prediction</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("📅 Age", 0, 100, 30)
    tumor_size = st.slider("🔍 Tumor Size (cm)", 1.0, 10.0, 5.0)
    tumor_grade = st.selectbox("🏥 Tumor Grade", [1, 2, 3])
    symptoms_severity = st.selectbox("🩺 Symptoms Severity", [1, 2, 3])

with col2:
    smoking_history = st.radio("🚬 Smoking History", ["Non-Smoker", "Former Smoker", "Current Smoker"])
    alcohol_consumption = st.selectbox("🍷 Alcohol Consumption", ["None", "Low", "Moderate", "High"])
    exercise_frequency = st.selectbox("🏃‍♂️ Exercise Frequency", ["Never", "Rarely", "Occasionally", "Regularly"])
    gender = st.radio("🚻 Gender", ["Male", "Female"])
    family_history = st.radio("👨‍👩‍👦 Family History", ["No", "Yes"])

# Encoding Categorical Inputs
smoking_dict = {"Non-Smoker": 0, "Former Smoker": 1, "Current Smoker": 2}
alcohol_dict = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
exercise_dict = {"Never": 0, "Rarely": 1, "Occasionally": 2, "Regularly": 3}
gender_dict = {"Male": 0, "Female": 1}
family_history_dict = {"No": 0, "Yes": 1}

input_data = [[
    age, tumor_size, tumor_grade, symptoms_severity, 
    smoking_dict[smoking_history], alcohol_dict[alcohol_consumption], 
    exercise_dict[exercise_frequency], gender_dict[gender], family_history_dict[family_history]
]]

# Prediction Button
if st.button("🔍 Predict Cancer Presence", key="predict_button"):
    if 'trained_model' in st.session_state:
        model = st.session_state['trained_model']
        input_df = pd.DataFrame(input_data, columns=X_train.columns)
        input_transformed = model.named_steps['preprocessor'].transform(input_df)
        prediction = model.named_steps['classifier'].predict(input_transformed)
        
        if prediction[0] == 1:
            st.markdown('<div class="alert-box">⚠️ Urgent Alert! Cancer Detected! <br> Please Consult a Doctor Immediately! 🏥</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="safe">✅ No Cancer Detected! Stay Healthy! 🏥</p>', unsafe_allow_html=True)
    else:
        st.error("❌ Please train a model first!")   
