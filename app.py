import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Sleep Disorder Analytics & Prediction",
    page_icon="🌙",
    layout="wide"
)

# Title & Description
st.title("🌙 Sleep Health & Lifestyle Analytics Dashboard")
st.markdown("""
This dashboard provides exploratory visual analytics on sleep habits and lifestyle metrics, 
paired with an interactive **Random Forest Machine Learning Predictor** for Sleep Disorders.
""")

# Load and Preprocess Data
@st.cache_data
def load_data():
    df = pd.read_csv('Sleep_disorder_data.csv')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df_clean = df.drop(columns=['Person ID', 'Blood Pressure'])
    return df_clean

df = load_data()

# Train Random Forest Model
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['Sleep Disorder'])
    y = data['Sleep Disorder']
    X_encoded = pd.get_dummies(X, columns=['Gender', 'Occupation', 'BMI Category'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_encoded.columns

model, feature_cols = train_model(df)

# Tabs Structure
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Interactive Prediction", "📁 Raw Data"])

# Tab 1: Exploratory Data Analysis
with tab1:
    st.header("Exploratory Data Analysis")
    
    # Key Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Participants", len(df))
    col2.metric("Avg Sleep Duration", f"{df['Sleep Duration'].mean():.2f} hrs")
    col3.metric("Avg Quality Score", f"{df['Quality of Sleep'].mean():.2f} / 10")
    col4.metric("Avg Stress Level", f"{df['Stress Level'].mean():.2f} / 10")
    
    st.divider()
    
    # Visualizations Grid
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sleep Disorder Distribution by BMI Category")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='BMI Category', hue='Sleep Disorder', palette='Set2', ax=ax)
        st.pyplot(fig)
        
    with c2:
        st.subheader("Sleep Disorder Prevalence by Occupation")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, y='Occupation', hue='Sleep Disorder', palette='Set1', ax=ax)
        st.pyplot(fig)
        
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                    'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    with c4:
        st.subheader("Quality of Sleep across Stress Levels")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Stress Level', y='Quality of Sleep', hue='Gender', palette='Blues', ax=ax)
        st.pyplot(fig)

# Tab 2: Prediction Interface
with tab2:
    st.header("Sleep Disorder Diagnosis Predictor")
    st.write("Input patient health and lifestyle attributes to estimate sleep disorder probability.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        gender = st.selectbox("Gender", df['Gender'].unique())
        age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 35)
        occupation = st.selectbox("Occupation", df['Occupation'].unique())
        bmi_cat = st.selectbox("BMI Category", df['BMI Category'].unique())
        sleep_dur = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, step=0.1)
        
    with col_b:
        sleep_qual = st.slider("Quality of Sleep (scale 1-10)", 1, 10, 7)
        activity_lvl = st.slider("Physical Activity Level (mins/day)", 0, 120, 60)
        stress_lvl = st.slider("Stress Level (scale 1-10)", 1, 10, 5)
        heart_rate = st.slider("Heart Rate (bpm)", 50, 100, 70)
        daily_steps = st.slider("Daily Steps", 1000, 15000, 7000, step=500)
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", 90, 180, 120)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 60, 120, 80)

    # Format inputs into model frame
    input_dict = {
        'Age': age,
        'Sleep Duration': sleep_dur,
        'Quality of Sleep': sleep_qual,
        'Physical Activity Level': activity_lvl,
        'Stress Level': stress_lvl,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic,
        'Gender': gender,
        'Occupation': occupation,
        'BMI Category': bmi_cat
    }
    
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df, columns=['Gender', 'Occupation', 'BMI Category'])
    
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    st.divider()
    if st.button("🔍 Predict Sleep Disorder Risk", use_container_width=True):
        prediction = model.predict(input_encoded)[0]
        probabilities = model.predict_proba(input_encoded)[0]
        classes = model.classes_
        
        if prediction == 'None':
            st.success("### Predicted Status: Low Risk (`None`)")
        elif prediction == 'Insomnia':
            st.warning("### Predicted Status: Elevated Risk for `Insomnia`")
        else:
            st.error("### Predicted Status: Elevated Risk for `Sleep Apnea`")
            
        st.subheader("Prediction Probabilities Breakdown")
        prob_df = pd.DataFrame({'Condition': classes, 'Probability': probabilities})
        st.bar_chart(prob_df.set_index('Condition'))

# Tab 3: Dataset Inspection
with tab3:
    st.header("Cleaned Dataset Inspection")
    st.dataframe(df, use_container_width=True)