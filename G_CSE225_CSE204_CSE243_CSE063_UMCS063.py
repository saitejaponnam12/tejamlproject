import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Hybrid Predictive Maintenance", layout="wide")

st.title("🔧 Hybrid Predictive Maintenance using NASA CMAPSS (FD001)")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train_FD001.txt", sep=" ", header=None)
    df.drop([26, 27], axis=1, inplace=True)
    df.columns = ['unit', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    return df

df = load_data()

# Clustering degradation stages
features = [col for col in df.columns if 'sensor' in col]
kmeans = KMeans(n_clusters=5, random_state=42)
df['stage'] = kmeans.fit_predict(df[features])

# Sidebar
st.sidebar.header("Select Engine Unit")
unit_ids = sorted(df['unit'].unique())
selected_unit = st.sidebar.selectbox("Engine ID", unit_ids)

# Show cycle data for selected unit
unit_df = df[df['unit'] == selected_unit]
st.subheader(f"📊 Engine {selected_unit} Degradation Overview")
st.line_chart(unit_df.set_index('cycle')[features[:3]])  # show first 3 sensors

# Classification
X = df[features]
y = df['stage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.subheader("📋 Stage Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# Regression - RUL Prediction
rul_df = df.groupby('unit')['cycle'].max().reset_index()
rul_df.columns = ['unit', 'max_cycle']
df = df.merge(rul_df, on='unit')
df['RUL'] = df['max_cycle'] - df['cycle']
df.drop(['max_cycle'], axis=1, inplace=True)

X_reg = df[features]
y_reg = df['RUL']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

st.subheader("📈 RUL Regression Metrics")
st.write(f"**RMSE:** {rmse:.2f} cycles")
st.write(f"**R² Score:** {r2:.2f}")

# Risk Alerts
st.subheader("⚠️ Risk Alerts")

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
y_pred_rul = reg.predict(X_test)

alerts = []
for i in range(len(X_test)):
    predicted_class = y_pred_class[i]
    prob_severe = y_pred_prob[i][3] + y_pred_prob[i][4]
    if predicted_class >= 3 and y_pred_rul[i] < 20 and prob_severe > 0.5:
        alerts.append({
            "Index": i,
            "Predicted Class": predicted_class,
            "RUL": round(y_pred_rul[i], 1),
            "Risk Score": round(prob_severe, 2)
        })

if alerts:
    st.dataframe(pd.DataFrame(alerts))
else:
    st.success("✅ No high-risk engines detected.")

# Optional Confusion Matrix
st.subheader("🧾 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)