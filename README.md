# Spotify Song Success Predictor

### UC Berkeley INDENG 242A - Final Project (Fall 2025)

This repository contains the source code and documentation for the final project of **INDENG 242A: Machine Learning and Data Analytics I**. The project focuses on predicting song popularity on Spotify using audio features and supervised machine learning techniques.

**🔗 Live Demo:** [Launch Dashboard](https://242a-spotify-dashboard.streamlit.app/)

## Project Overview

The goal of this project is to determine whether a song will be "successful" (defined as having a popularity score above the median) based on its acoustic characteristics such as danceability, energy, and valence.

We implemented a **Random Forest Classifier** which demonstrated superior performance in modeling non-linear relationships compared to logistic regression baselines.

## Dashboard Features

We developed an interactive web application using **Streamlit** to visualize our findings and deploy the model:

* **Exploratory Data Analysis (EDA):** Interactive correlation heatmaps and box plots comparing successful vs. unsuccessful songs.
* **Model Performance:** Visualization of feature importance and performance metrics (Accuracy, AUC).
* **Prediction Playground:** An interactive interface allowing users to adjust audio feature sliders (e.g., Tempo, Loudness) to simulate a song and receive a real-time success prediction.

## Data Source

The dataset used in this project is sourced from Kaggle:
[Spotify Music Dataset by Solomon Ameen](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)

## Installation & Local Usage

To run the dashboard locally:

1. Clone this repository.

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run dashboard.py
   ```

## Contributors

* Yijun Gu
* Rimsha Ijaz
* Yizhou Zheng

---

*Created for INDENG 242A, Department of Industrial Engineering & Operations Research, UC Berkeley.*
