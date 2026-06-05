import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

from scipy.spatial.distance import cdist

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ==================================================
# THEME
# ==================================================

st.markdown("""
<style>

.stApp{
    background-color:#F1F8F4;
}

/* Headers */
h1,h2,h3,h4{
    color:#1B4332 !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#D8F3DC;
}

/* Text visibility */
p, span, label{
    color:#1B4332 !important;
}

/* Metric cards */
div[data-testid="metric-container"]{
    background-color:white;
    border-radius:15px;
    padding:15px;
    border:1px solid #95D5B2;
}

/* Buttons */
.stButton > button{
    background-color:#2D6A4F;
    color:white;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/Wholesale customers data.csv"
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Configuration")

cluster_type = st.sidebar.radio(
    "Hierarchical Method",
    [
        "Agglomerative",
        
    ]
)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    [
        "ward",
        "complete",
        "average",
        "single"
    ]
)

num_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    4
)
# ==================================================
# TITLE
# ==================================================

st.title("🌳 Hierarchical Clustering Dashboard")

st.markdown("""
Explore customer groups using Hierarchical Clustering.
""")

# ==================================================
# OVERVIEW
# ==================================================
st.subheader("📊 Dataset Overview")

c1,c2,c3 = st.columns(3)

with c1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with c2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with c3:
    st.metric(
        "Clusters",
        num_clusters
    )

# ==================================================
# DATA PREVIEW
# ==================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)


st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(10,5)
)

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="Greens",
    ax=ax
)

st.pyplot(fig)


features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


st.subheader("🌿 Dendrogram")

linked = linkage(
    X_scaled,
    method=linkage_method
)

fig, ax = plt.subplots(
    figsize=(12,5)
)

dendrogram(linked)

st.pyplot(fig)


model = AgglomerativeClustering(
    n_clusters=num_clusters,
    linkage=linkage_method
)

clusters = model.fit_predict(
    X_scaled
)


# ==================================================
# PCA VISUALIZATION
# ==================================================

st.subheader("🎯 Cluster Visualization")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

ax.set_title(
    f"{cluster_type} Clustering"
)

st.pyplot(fig)

# ==================================================
# SILHOUETTE SCORE
# ==================================================

score = silhouette_score(
    X_scaled,
    clusters
)

st.success(
    f"⭐ Silhouette Score: {score:.3f}"
)
# ==================================================
# CLUSTER EXPLORER
# ==================================================

st.subheader("🔍 Cluster Explorer")

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(set(clusters))
)

cluster_size = np.sum(
    clusters == selected_cluster
)

st.info(
    f"Cluster {selected_cluster} contains {cluster_size} customers"
)
# ==================================================
# FEATURE DISTRIBUTION
# ==================================================

st.subheader("📈 Feature Distribution")

feature = st.selectbox(
    "Choose Feature",
    df.columns
)

fig, ax = plt.subplots()

sns.histplot(
    df[feature],
    kde=True,
    color="green",
    ax=ax
)

st.pyplot(fig)



# ==================================================
# CUSTOMER CLUSTER PREDICTION
# ==================================================

st.subheader("🤖 Customer Cluster Prediction")

col1, col2 = st.columns(2)

with col1:

    fresh = st.slider(
        "Fresh",
        int(df["Fresh"].min()),
        int(df["Fresh"].max()),
        int(df["Fresh"].mean())
    )

    milk = st.slider(
        "Milk",
        int(df["Milk"].min()),
        int(df["Milk"].max()),
        int(df["Milk"].mean())
    )

    grocery = st.slider(
        "Grocery",
        int(df["Grocery"].min()),
        int(df["Grocery"].max()),
        int(df["Grocery"].mean())
    )

with col2:

    frozen = st.slider(
        "Frozen",
        int(df["Frozen"].min()),
        int(df["Frozen"].max()),
        int(df["Frozen"].mean())
    )

    detergents = st.slider(
        "Detergents_Paper",
        int(df["Detergents_Paper"].min()),
        int(df["Detergents_Paper"].max()),
        int(df["Detergents_Paper"].mean())
    )

    delicassen = st.slider(
        "Delicassen",
        int(df["Delicassen"].min()),
        int(df["Delicassen"].max()),
        int(df["Delicassen"].mean())
    )

if st.button("Predict Cluster"):

    # User Input
    user_data = pd.DataFrame(
        [[
            fresh,
            milk,
            grocery,
            frozen,
            detergents,
            delicassen
        ]],
        columns=features
    )

    # Scale Input
    user_scaled = scaler.transform(user_data)

    # Calculate Cluster Centers
    centers = []

    for cluster_id in sorted(np.unique(clusters)):

        center = X_scaled[
            clusters == cluster_id
        ].mean(axis=0)

        centers.append(center)

    centers = np.array(centers)

    # Distance Calculation
    distances = cdist(
        user_scaled,
        centers
    )

    predicted_cluster = np.argmin(distances)

    st.success(
        f"🎯 Predicted Cluster : {predicted_cluster}"
    )

    st.info(
        f"""
        Customer is closest to
        Cluster {predicted_cluster}
        based on hierarchical clustering.
        """
    )

    # Show Cluster Size

    cluster_size = np.sum(
        clusters == predicted_cluster
    )

    st.metric(
        "Customers in Cluster",
        cluster_size
    )