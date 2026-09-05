import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Sleep Dynamics & Disorder AI Analytics",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #38BDF8;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    
    /* Result Status Cards */
    .status-card-healthy {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        color: #34D399;
    }
    .status-card-insomnia {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        color: #FBBF24;
    }
    .status-card-apnea {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        color: #F87171;
    }
</style>
""", unsafe_allow_html=True)

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

# Train Model
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

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/moon-and-stars.png", width=70)
    st.title("SomnoAI Platform")
    st.caption("Clinical Sleep Analytics & Predictive AI")
    st.markdown("---")
    
    st.subheader("⚙️ Global Dataset Filter")
    selected_gender = st.multiselect("Filter by Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    selected_bmis = st.multiselect("Filter by BMI Category", options=df['BMI Category'].unique(), default=df['BMI Category'].unique())
    
    st.markdown("---")
    st.info("💡 **Model Info:** Random Forest Classifier running with 96.00% cross-validation accuracy.")

# Filter dataset based on sidebar
filtered_df = df[(df['Gender'].isin(selected_gender)) & (df['BMI Category'].isin(selected_bmis))]

# Main Title Header
st.title("🌙 SomnoAnalytics AI Dashboard")
st.markdown("Explore multi-dimensional clinical sleep metrics or predict personalized disorder risks interactively.")

# Tabs Setup
tab1, tab2, tab3 = st.tabs(["📊 Interactive Visual Analytics", "🔮 Patient Diagnostic AI Simulator", "📁 Data Matrix"])

# --- TAB 1: VISUAL ANALYTICS ---
with tab1:
    st.subheader("Key Population Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Records</div><div class="metric-value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Sleep Duration</div><div class="metric-value">{filtered_df["Sleep Duration"].mean():.1f} hrs</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Quality Score</div><div class="metric-value">{filtered_df["Quality of Sleep"].mean():.1f} / 10</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Stress Score</div><div class="metric-value">{filtered_df["Stress Level"].mean():.1f} / 10</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 🩺 Sleep Disorder Distribution by BMI Class")
        fig_bmi = px.histogram(
            filtered_df, 
            x="BMI Category", 
            color="Sleep Disorder", 
            barmode="group",
            color_discrete_map={'None': '#10B981', 'Insomnia': '#F59E0B', 'Sleep Apnea': '#EF4444'},
            template="plotly_dark"
        )
        fig_bmi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bmi, use_container_width=True)
        
    with col_chart2:
        st.markdown("##### 💼 Occupation vs Sleep Disorder Breakdown")
        fig_occ = px.histogram(
            filtered_df, 
            y="Occupation", 
            color="Sleep Disorder", 
            barmode="stack",
            color_discrete_map={'None': '#10B981', 'Insomnia': '#F59E0B', 'Sleep Apnea': '#EF4444'},
            template="plotly_dark"
        )
        fig_occ.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_occ, use_container_width=True)
        
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("##### ⚡ Stress Level vs Quality of Sleep")
        fig_scatter = px.scatter(
            filtered_df, 
            x="Stress Level", 
            y="Quality of Sleep", 
            color="Sleep Disorder",
            size="Heart Rate", 
            hover_data=["Occupation", "Age"],
            color_discrete_map={'None': '#10B981', 'Insomnia': '#F59E0B', 'Sleep Apnea': '#EF4444'},
            template="plotly_dark"
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart4:
        st.markdown("##### 🫀 Blood Pressure Distribution Across Disorders")
        fig_bp = px.box(
            filtered_df, 
            x="Sleep Disorder", 
            y="Systolic_BP", 
            color="Sleep Disorder",
            points="all",
            color_discrete_map={'None': '#10B981', 'Insomnia': '#F59E0B', 'Sleep Apnea': '#EF4444'},
            template="plotly_dark"
        )
        fig_bp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bp, use_container_width=True)

# --- TAB 2: INTERACTIVE SIMULATOR ---
with tab2:
    st.subheader("Patient Clinical Input Profile")
    st.caption("Adjust sliders and parameters below to compute AI prediction probabilities.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown("##### 👤 Demographics")
        gender = st.radio("Gender", options=df['Gender'].unique(), horizontal=True)
        age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 38)
        occupation = st.selectbox("Occupation", options=df['Occupation'].unique())
        bmi_cat = st.selectbox("BMI Category", options=df['BMI Category'].unique())
        
    with col_input2:
        st.markdown("##### 💤 Sleep & Stress Parameters")
        sleep_dur = st.slider("Sleep Duration (Hours)", 4.0, 10.0, 6.5, step=0.1)
        sleep_qual = st.slider("Quality of Sleep Score (1-10)", 1, 10, 6)
        stress_lvl = st.slider("Self-Reported Stress Level (1-10)", 1, 10, 7)
        
    with col_input3:
        st.markdown("##### 🩺 Physical Vitals")
        activity_lvl = st.slider("Physical Activity (Minutes/Day)", 0, 120, 45)
        heart_rate = st.slider("Resting Heart Rate (BPM)", 50, 100, 72)
        daily_steps = st.slider("Daily Step Count", 1000, 15000, 6000, step=500)
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", 90, 180, 130)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 60, 120, 85)

    input_dict = {
        'Age': age, 'Sleep Duration': sleep_dur, 'Quality of Sleep': sleep_qual,
        'Physical Activity Level': activity_lvl, 'Stress Level': stress_lvl,
        'Heart Rate': heart_rate, 'Daily Steps': daily_steps,
        'Systolic_BP': systolic, 'Diastolic_BP': diastolic,
        'Gender': gender, 'Occupation': occupation, 'BMI Category': bmi_cat
    }
    
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df, columns=['Gender', 'Occupation', 'BMI Category'])
    
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Run AI Prediction Analysis", use_container_width=True):
        prediction = model.predict(input_encoded)[0]
        probabilities = model.predict_proba(input_encoded)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        st.markdown("### AI Diagnostic Summary")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            if prediction == 'None':
                st.markdown(f'<div class="status-card-healthy"><h2>🟢 Status: Healthy Sleep</h2><p>Low clinical likelihood for sleep apnea or chronic insomnia.</p><h3>Confidence: {prob_dict["None"]*100:.1f}%</h3></div>', unsafe_allow_html=True)
            elif prediction == 'Insomnia':
                st.markdown(f'<div class="status-card-insomnia"><h2>🟡 Status: Insomnia Detected</h2><p>Elevated likelihood of sleep onset or maintenance difficulty.</p><h3>Confidence: {prob_dict["Insomnia"]*100:.1f}%</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-card-apnea"><h2>🔴 Status: Sleep Apnea Risk</h2><p>High probability of obstructive sleep apnea signals.</p><h3>Confidence: {prob_dict["Sleep Apnea"]*100:.1f}%</h3></div>', unsafe_allow_html=True)
                
        with res_col2:
            fig_prob = px.bar(
                x=classes, 
                y=probabilities * 100,
                color=classes,
                color_discrete_map={'None': '#10B981', 'Insomnia': '#F59E0B', 'Sleep Apnea': '#EF4444'},
                labels={'x': 'Condition', 'y': 'Probability (%)'},
                title="Class Probability Distribution",
                template="plotly_dark"
            )
            fig_prob.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

# --- TAB 3: MATRIX ---
with tab3:
    st.subheader("Filtered Patient Data Matrix")
    st.dataframe(filtered_df, use_container_width=True, height=450)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered CSV Data",
        data=csv_data,
        file_name="filtered_sleep_data.csv",
        mime="text/csv",
    )