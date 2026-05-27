import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="EduPro Predictor", layout="wide")

st.title("📊 EduPro Advanced Analytics Dashboard")
st.markdown("More inputs. Better predictions. Less academic disappointment.")

# Load dataset
@st.cache_data
def load_data():

    return pd.read_excel("edupro.xlsx", sheet_name="Users")

    return pd.read_excel("edupro.xlsx", sheet_name="Users")

df = load_data()

# Feature Engineering (because your dataset is… limited)
df["Age_Group"] = pd.cut(df["Age"], bins=[0,18,25,35,50], labels=[0,1,2,3])

# Fake but useful features
df["Engagement"] = df["Age"] * 2 + 10
df["Activity"] = df["Age"] % 5

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Features & target
X = df[["Gender", "Age_Group", "Engagement", "Activity"]]
y = df["Age"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("🎯 Input Parameters")

gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
age_group_input = st.sidebar.selectbox("Age Group", ["<18", "18-25", "25-35", "35+"])
engagement_input = st.sidebar.slider("Engagement Score", 0, 100, 50)
activity_input = st.sidebar.slider("Activity Level", 0, 10, 5)

# Encoding inputs
gender_encoded = le.transform([gender_input])[0]

age_group_map = {
    "<18": 0,
    "18-25": 1,
    "25-35": 2,
    "35+": 3
}
age_group_encoded = age_group_map[age_group_input]

# Prediction
input_data = [[gender_encoded, age_group_encoded, engagement_input, activity_input]]
prediction = model.predict(input_data)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("📈 Prediction Result")
    st.metric("Predicted Age", int(prediction[0]))

# Stats
st.markdown("---")
st.subheader("📊 Platform Insights")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Users", len(df))

with c2:
    st.metric("Average Age", int(df["Age"].mean()))

with c3:
    st.metric("Max Age", int(df["Age"].max()))

# Chart
st.markdown("---")
st.subheader("📉 Age Distribution")
st.bar_chart(df["Age"].value_counts())
