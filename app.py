import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import plotly.express as px
import plotly.graph_objects as go

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Mall Customer GMM Clustering",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Mall Customer Segmentation using Gaussian Mixture Model")

# ==================================
# LOAD DATA
# ==================================
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# ==================================
# FEATURES
# ==================================
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

# ==================================
# SCALING
# ==================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================
# GMM MODEL
# ==================================
n_clusters = 5

gmm = GaussianMixture(
    n_components=n_clusters,
    random_state=42
)

clusters = gmm.fit_predict(X_scaled)

df["Cluster"] = clusters

# ==================================
# METRICS
# ==================================
c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", len(df))
c2.metric("Clusters", n_clusters)
c3.metric("Features Used", len(features))

# ==================================
# USER INPUT
# ==================================
st.subheader("Predict Customer Segment")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)

    income = st.slider(
        "Annual Income (k$)",
        10,
        150,
        60
    )

with col2:
    score = st.slider(
        "Spending Score (1-100)",
        1,
        100,
        50
    )

# ==================================
# PREDICT
# ==================================
if st.button("Predict Cluster"):

    customer = pd.DataFrame({
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [score]
    })

    customer_scaled = scaler.transform(customer)

    cluster = gmm.predict(customer_scaled)[0]

    probabilities = gmm.predict_proba(customer_scaled)[0]

    confidence = np.max(probabilities) * 100

    st.success(
        f"Predicted Cluster: {cluster}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

# ==================================
# 3D VISUALIZATION
# ==================================
st.subheader("3D Cluster Visualization")

fig = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color=df["Cluster"].astype(str),
    hover_data=["CustomerID"]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================
# CLUSTER DISTRIBUTION
# ==================================
st.subheader("Cluster Distribution")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

fig2 = go.Figure(
    data=[
        go.Pie(
            labels=cluster_counts.index.astype(str),
            values=cluster_counts.values
        )
    ]
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==================================
# CLUSTER STATISTICS
# ==================================
st.subheader("Cluster Summary")

summary = (
    df.groupby("Cluster")[
        ["Age",
         "Annual Income (k$)",
         "Spending Score (1-100)"]
    ]
    .mean()
    .round(2)
)

st.dataframe(summary)

# ==================================
# DATASET PREVIEW
# ==================================
st.subheader("Clustered Dataset")

st.dataframe(df)