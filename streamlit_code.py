import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("Heart Attack Analysis and Prediction")
st.markdown("""
This interactive dashboard provides an exploratory analysis of the heart attack dataset.
It includes visualizations, cleaned data, statistical insights, and machine learning model predictions.
""")

if "show_details" not in st.session_state:
    st.session_state.show_details = False


if st.button("Show/Hide Project Info"):
    st.session_state.show_details = not st.session_state.show_details

if st.session_state.show_details:
    st.subheader("Project Justification")
    st.markdown("""
    Heart diseases are one of the leading causes of mortality worldwide. This project is built to help
    identify risk factors for heart attacks and provide predictive insights that can aid medical professionals 
    in decision-making. The dashboard leverages machine learning models to predict heart attack likelihood 
    and helps analyze patient data interactively.
    """)


@st.cache_data
def load_data():
    conn = sqlite3.connect("/Users/jeeva/Documents/heart_attack_prediction.db")

    query = """
    SELECT p.age, p.sex, p.age_group, m.cp, m.trtbps, m.chol, m.fbs, m.restecg, 
           m.thalachh, m.exng, m.oldpeak, m.slp, m.caa, m.thall, r.output
    FROM patients p
    JOIN medical_data m ON p.id = m.id
    JOIN results r ON p.id = r.id;
    """
    data = pd.read_sql(query, conn)
    conn.close()
    return data


data = load_data()
if st.checkbox("Show Data"):
    st.write(data)

# Dashboard (visualization code)

visualization = st.radio(
    "Choose a Visualization:",
    ["Age Distribution", "Cholesterol vs Age", "Correlation Heatmap", "Pairplot of Selected Features","Average Blood Pressure by Age Group"]
)

# Age Distribution
if visualization == "Age Distribution":
    st.subheader("Age Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['age'], bins=10, kde=True)
    plt.title("Age Distribution")
    st.pyplot(plt)
    st.markdown("Age Distribution: The dataset includes individuals across a broad age range, with a concentration in middle-aged and senior groups (40–70 years). This suggests heart disease risks are evaluated for a high-risk demographic.")

# Cholesterol vs Age
elif visualization == "Cholesterol vs Age":
    st.subheader("Cholesterol vs Age")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['age'], y=data['chol'], hue=data['output'], palette='coolwarm')
    plt.title("Cholesterol vs Age")
    st.pyplot(plt)
    st.markdown("Age vs. Cholesterol (Scatter Plot): Older individuals tend to have higher cholesterol levels. Among these, patients with heart attacks (output = 1) often have elevated cholesterol compared to those without.")

# Correlation Heatmap
elif visualization == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
    st.markdown("Positive Correlations-thalachh (maximum heart rate achieved) is negatively correlated with heart attack occurrence, indicating that individuals with lower max heart rates are more likely to have heart issues. oldpeak (ST depression) has a strong positive correlation with heart attack outcomes, highlighting its diagnostic significance. cp (chest pain type) shows a moderately positive correlation with heart attack outcomes. Negative Correlations: thalachh (maximum heart rate achieved) has a strong negative correlation with the heart attack outcome (output), indicating it is a critical feature.")

# Pairplot of Selected Features
elif visualization == "Pairplot of Selected Features":
    st.subheader("Pairplot of Selected Features")

    selected_features = ['age', 'chol', 'trtbps', 'thalachh', 'output']

    plt.figure()
    pairplot = sns.pairplot(data[selected_features], hue='output', diag_kind='kde', palette='coolwarm')
    st.pyplot(pairplot.fig) 
    st.markdown("""
    This pairplot visualizes relationships among features like age, cholesterol (chol), 
    resting blood pressure (trtbps), maximum heart rate achieved (thalachh), and heart attack outcomes.
    """)
# Average Blood Pressure by Age Group
elif visualization == "Average Blood Pressure by Age Group":
    st.subheader("Average Blood Pressure by Age Group")
    plt.figure(figsize=(8, 6))
    sns.barplot(x='age_group', y='trtbps', data=data, errorbar=None, color='lightblue')
    plt.title("Average Blood Pressure by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Average Blood Pressure (mmHg)")
    st.pyplot(plt)
    
    st.markdown("""
    **Insight**:  
    The bar chart shows that average blood pressure increases with age, with the 50-60 
    and 60+ age groups having the highest values.
    """)      

# Machine Learning code

# Sidebar for prediction inputs
st.sidebar.title("Heart Attack Prediction")
st.sidebar.markdown("Provide the following details to predict the likelihood of a heart attack:")


# Input fields for user
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trtbps = st.sidebar.number_input("Resting Blood Pressure (trtbps)", min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol (chol)", min_value=100, max_value=500, value=200)
thalachh = st.sidebar.number_input("Max Heart Rate Achieved (thalachh)", min_value=50, max_value=250, value=150)
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)


user_input = np.array([[age, 1 if sex == "Male" else 0, cp, trtbps, chol, thalachh, oldpeak]])

predictors = ['age', 'sex', 'cp', 'trtbps', 'chol', 'thalachh', 'oldpeak']
X = data[predictors]
y = data['output']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
user_input_scaled = scaler.transform(user_input)

# Train the KNN model
@st.cache_resource
def train_knn(X, y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn

knn_model = train_knn(X_scaled, y)

# Make a prediction
prediction = knn_model.predict(user_input_scaled)[0]
prediction_proba = knn_model.predict_proba(user_input_scaled)[0]

# Display Prediction Result
st.subheader("Prediction Result")
if prediction == 1:
    st.success(f"The model predicts a HIGH likelihood of a heart attack.")
else:
    st.info(f"The model predicts a LOW likelihood of a heart attack.")

