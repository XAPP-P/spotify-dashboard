import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

st.set_page_config(page_title="Spotify Success Predictor", layout="wide")

# Define the local paths as requested
PATH_HIGH = "high_popularity_spotify_data.csv"
PATH_LOW = "low_popularity_spotify_data.csv"

@st.cache_data
def load_and_process_data():
    """
    Loads data from local paths, combines them, handles duplicates,
    and creates the binary target variable based on the report's methodology.
    """
    try:
        # Load datasets
        df_high = pd.read_csv(PATH_HIGH)
        df_low = pd.read_csv(PATH_LOW)
        
        # Combine datasets
        df = pd.concat([df_high, df_low], axis=0, ignore_index=True)
        
        # Drop duplicates based on track_id if it exists, otherwise strict duplicate check
        if 'track_id' in df.columns:
            df = df.drop_duplicates(subset=['track_id'])
        else:
            df = df.drop_duplicates()
            
        # --- Feature Engineering (Based on Project Draft) ---
        # "Song success is defined using a popularity threshold, denoted by τ, 
        # which is set to the median popularity score in the dataset."
        
        median_popularity = df['track_popularity'].median()
        
        # Create binary target: 1 = Successful, 0 = Unsuccessful
        df['is_successful'] = (df['track_popularity'] >= median_popularity).astype(int)
        
        return df, median_popularity
        
    except FileNotFoundError as e:
        st.error(f"Error loading files. Please check the paths:\n{e}")
        return pd.DataFrame(), 0

# Load data
df, median_threshold = load_and_process_data()

if df.empty:
    st.stop()

# Define features used for modeling (Numeric Audio Features)
features_list = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]

# Clean data for modeling (drop NaNs in selected features)
df_model = df[features_list + ['is_successful']].dropna()

# ==========================================
# 2. MODEL TRAINING (Random Forest)
# ==========================================

@st.cache_resource
def train_model(data):
    """
    Trains a Random Forest model on the fly to support the interactive predictor.
    Based on the report finding that RF performs better than Logistic Regression.
    """
    X = data[features_list]
    y = data['is_successful']
    
    # Split data (80/20 split as typical in the course)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest (Parameters similar to teammates' code)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Generate predictions for metrics
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    return rf, acc, auc, X_test, y_test

model, acc, auc, X_test, y_test = train_model(df_model)

# ==========================================
# 3. DASHBOARD LAYOUT
# ==========================================

# Title and Context
st.title("🎵 Spotify Song Success Dashboard")
st.markdown(f"""
**Project:** INDENG 242A Final Project  
**Objective:** Predict song popularity based on audio features.  
**Definition of Success:** Popularity Score $\ge$ {int(median_threshold)} (Median).
""")

# Create Tabs for organized viewing
tab1, tab2, tab3 = st.tabs(["📊 Data Insights (EDA)", "PY Model Performance", "🎧 Prediction Playground"])

# --- TAB 1: DATA EXPLORATORY ANALYSIS ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Songs", len(df))
    col2.metric("Successful Songs", len(df[df['is_successful'] == 1]))
    col3.metric("Unsuccessful Songs", len(df[df['is_successful'] == 0]))
    
    st.divider()
    
    # Interactive Correlation Heatmap
    st.subheader("Feature Correlation Matrix")
    st.markdown("This heatmap reveals relationships between audio features and success.")
    
    corr_cols = features_list + ['track_popularity', 'is_successful']
    corr_matrix = df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.divider()
    
    # Comparing Distributions
    st.subheader("Audio Features: Successful vs. Unsuccessful")
    st.markdown("Select a feature to compare its distribution between the two classes.")
    
    selected_feat = st.selectbox("Select Feature", features_list, index=0)
    
    # Box Plot Comparison
    fig_box = px.box(
        df, 
        x="is_successful", 
        y=selected_feat, 
        color="is_successful",
        labels={"is_successful": "Is Successful (0=No, 1=Yes)"},
        title=f"Distribution of {selected_feat} by Success Class"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.header("Random Forest Model Evaluation")
    st.markdown("As noted in the report, **Random Forest** showed superior ability to model non-linear relationships compared to Logistic Regression.")
    
    # Metrics
    m_col1, m_col2 = st.columns(2)
    m_col1.info(f"**Test Set Accuracy:** {acc:.2%}")
    m_col2.success(f"**Test Set AUC Score:** {auc:.4f}")
    
    st.divider()
    
    # Feature Importance
    st.subheader("Feature Importance")
    st.markdown("Which audio features drive the model's predictions the most?")
    
    importances = pd.DataFrame({
        'Feature': features_list,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(
        importances, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title="Random Forest Feature Importance",
        color='Importance'
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 3: INTERACTIVE PREDICTION ---
with tab3:
    st.header("Predict Song Success")
    st.markdown("Adjust the sliders below to simulate a new song's features. The model will predict if it's likely to be a 'Hit'.")
    
    # Input Form
    with st.form("prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            val_dance = st.slider("Danceability", 0.0, 1.0, 0.5)
            val_energy = st.slider("Energy", 0.0, 1.0, 0.5)
            val_valence = st.slider("Valence (Positivity)", 0.0, 1.0, 0.5)
            val_speech = st.slider("Speechiness", 0.0, 1.0, 0.05)
            
        with col_p2:
            val_acoustic = st.slider("Acousticness", 0.0, 1.0, 0.1)
            val_live = st.slider("Liveness", 0.0, 1.0, 0.1)
            val_instru = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
            val_tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0)
            
        with col_p3:
            val_loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0)
            val_duration = st.number_input("Duration (ms)", value=200000)
            val_key = st.selectbox("Key", range(12), index=0)
            val_mode = st.selectbox("Mode (1=Major, 0=Minor)", [0, 1], index=1)
            
        submit_button = st.form_submit_button("Run Prediction")
        
    if submit_button:
        # Prepare input array
        input_data = pd.DataFrame([[
            val_dance, val_energy, val_key, val_loudness, val_mode, 
            val_speech, val_acoustic, val_instru, val_live, 
            val_valence, val_tempo, val_duration
        ]], columns=features_list)
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"### 🎉 Prediction: SUCCESSFUL (Hit Song)")
            st.write(f"Confidence: **{probability:.2%}**")
        else:
            st.error(f"### 📉 Prediction: UNSUCCESSFUL")
            st.write(f"Confidence of Success: **{probability:.2%}**")
            
        st.info("Note: This prediction is based on the Random Forest model trained on the provided dataset.")

# Footer
st.markdown("---")
st.markdown("*Dashboard created for INDENG 242A Final Project.*")