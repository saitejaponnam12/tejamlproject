 Hybrid Predictive Maintenance using NASA CMAPSS (FD001)
Group Name: G_CSE225_CSE204_CSE243_CSE063_UMCS063
Project Title: Hybrid Predictive Maintenance using NASA CMAPSS Dataset
A Streamlit App for Engine Health Monitoring and RUL Prediction
 Objective:
This project predicts engine degradation stages and Remaining Useful Life (RUL) using
NASA’s CMAPSS dataset. It combines unsupervised clustering, classification, regression,
and risk alerting in an interactive web interface.
 Key Features:
• KMeans Clustering (5 Stages): Automatically labels engine health as Normal →
Slightly Degraded → Moderately Degraded → Critical → Failure.
• Interactive Dashboard: Select any engine to visualize its sensor readings over
time.
• Stage Classification: Uses RandomForestClassifier to classify current degradation
stage.
• RUL Prediction: Predicts how many cycles are left before failure using
RandomForestRegressor.
• Risk Alerts: Triggers alerts if RUL < 20 cycles & stage ≥ 3 with high probability.
• Confusion Matrix & Metrics: Visual performance reports using heatmaps and
scoring.
 Models Used:
• KMeans (for clustering stages)
• RandomForestClassifier (for stage classification)
• RandomForestRegressor (for RUL prediction)
 Dataset:
• Source: NASA CMAPSS FD001 (Kaggle Link)
• Contains sensor readings across engine cycles with no predefined labels.
 Tech Stack:
Python, Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn
▶ How to Run:
1. Install libraries (pip install -r requirements.txt)
2. Place train_FD001.txt in the root folder
3. Run with streamlit run app.py
