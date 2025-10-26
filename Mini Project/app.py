import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import io

# --- App Configuration ---
st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍",
    layout="wide"
)

# --- Abstract ---
APP_ABSTRACT = """
Customer segmentation is a vital strategy in modern retail analytics, helping businesses understand and target their customers effectively. This project, Customer Segmentation for Retail Using K-Means and PCA, aims to group customers based on purchasing patterns and demographic attributes to support data-driven marketing decisions. By identifying segments such as high-income high-spenders or budget-conscious customers, retailers can design personalized offers and improve overall customer satisfaction.

The project employs the K-Means clustering algorithm to discover natural customer groupings and uses Principal Component Analysis (PCA) for dimensionality reduction and visual interpretation of clusters. The dataset is preprocessed by handling missing values, encoding categorical data, and standardizing numerical features. The Elbow Method and Silhouette Score are applied to determine the optimal number of clusters, ensuring meaningful segmentation and strong cluster separation.

An interactive Streamlit application is developed to make the model accessible and user-friendly. The app allows users to upload data, adjust clustering parameters, visualize customer clusters in real time, and predict the segment of new customers. This project demonstrates how unsupervised learning can transform raw retail data into actionable business insights, enabling smarter marketing strategies and more efficient customer relationship management.
"""

# --- Helper Functions ---

@st.cache_data
def load_default_data():
    """Loads the default Mall Customers dataset."""
    csv_data = """CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14
26,Male,29,28,82
27,Female,45,28,32
28,Male,35,28,61
29,Female,40,29,31
30,Female,23,29,87
31,Male,60,30,4
32,Female,21,30,73
33,Male,53,33,4
34,Female,18,33,92
35,Female,49,33,14
36,Female,21,33,81
37,Female,42,34,17
38,Female,30,34,73
39,Female,36,37,26
40,Female,65,38,35
41,Female,20,38,92
42,Male,48,39,36
43,Female,24,39,65
44,Female,31,39,8
45,Female,49,39,28
46,Female,24,39,68
47,Female,50,40,55
48,Female,27,40,47
49,Female,29,40,42
50,Female,31,40,42
"""
    return pd.read_csv(io.StringIO(csv_data))

@st.cache_resource
def create_preprocessing_pipeline(df):
    """Creates a preprocessing pipeline based on dataframe columns."""
    
    # Identify numeric and categorical features (exclude CustomerID)
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if 'CustomerID' in numeric_features:
        numeric_features.remove('CustomerID')
        
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create the full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, numeric_features, categorical_features

@st.cache_data
def calculate_cluster_metrics(_processed_data):
    """Calculates inertia and silhouette scores for a range of k."""
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(_processed_data)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(processed_data, kmeans.labels)
        silhouette_scores.append(score)
        
    metrics_df = pd.DataFrame({
        'K': k_range,
        'Inertia (Elbow)': inertia,
        'Silhouette Score': silhouette_scores
    })
    return metrics_df

# --- Sidebar Controls ---
st.sidebar.title("🛍 Segmentation Controls")
st.sidebar.markdown("Configure the analysis parameters.")

uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using default 'Mall Customers' dataset. Upload a CSV to use your own data.")
    df = load_default_data()

# Drop CustomerID if it exists, as it's just an identifier
if 'CustomerID' in df.columns:
    df_analysis = df.drop(columns=['CustomerID'])
else:
    df_analysis = df.copy()

# Create pipeline and process data
try:
    preprocessor, numeric_cols, cat_cols = create_preprocessing_pipeline(df_analysis)
    processed_data = preprocessor.fit_transform(df_analysis)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Parameters")
    k_slider = st.sidebar.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Use the 'Optimal K Analysis' tab to find the best value for K."
    )
    
    # --- Main Application ---
    st.title("Retail Customer Segmentation Dashboard")
    st.info(APP_ABSTRACT)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview", 
        "Optimal K Analysis", 
        "Segmentation Results", 
        "Predict New Customer"
    ])

    # --- Tab 1: Data Overview ---
    with tab1:
        st.header("Data Preview")
        st.dataframe(df.head(), width='stretch')
        
        st.header("Data Statistics")
        st.dataframe(df.describe(), width='stretch')
        
        st.header("Features Identified")
        st.markdown(f"*Numeric Features:* {', '.join(numeric_cols)}")
        st.markdown(f"*Categorical Features:* {', '.join(cat_cols)}")

    # --- Tab 2: Optimal K Analysis ---
    with tab2:
        st.header("Finding the Optimal Number of Clusters (K)")
        st.markdown("""
        We use two methods to find the best 'K':
        1.  *Elbow Method:* Look for the "elbow" point where the inertia (sum of squared distances) stops decreasing rapidly.
        2.  *Silhouette Score:* Measures how similar an object is to its own cluster compared to other clusters. A higher score (closer to 1) is better.
        """)
        
        with st.spinner("Calculating cluster metrics for K=2 to 10..."):
            metrics_df = calculate_cluster_metrics(processed_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Elbow Method (Inertia)")
            fig_elbow = px.line(metrics_df, x='K', y='Inertia (Elbow)', title='Elbow Method', markers=True)
            st.plotly_chart(fig_elbow, use_container_width=True) # Note: Keeping use_container_width as 'width' is not a direct replacement for plotly
            
        with col2:
            st.subheader("Silhouette Score")
            fig_sil = px.line(metrics_df, x='K', y='Silhouette Score', title='Silhouette Score', markers=True)
            st.plotly_chart(fig_sil, use_container_width=True) # Note: Keeping use_container_width as 'width' is not a direct replacement for plotly

    # --- Tab 3: Segmentation Results ---
    with tab3:
        st.header(f"Customer Segments (K={k_slider})")
        
        # Run K-Means with the selected K
        kmeans_model = KMeans(n_clusters=k_slider, init='k-means++', n_init=10, random_state=42)
        kmeans_model.fit(processed_data)
        
        # Add cluster labels to the original dataframe
        df_results = df_analysis.copy()
        df_results['Cluster'] = kmeans_model.labels_.astype(str) # Convert to string for Plotly
        
        # Run PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(processed_data)
        
        df_results['PC1'] = pca_features[:, 0]
        df_results['PC2'] = pca_features[:, 1]
        
        # Create PCA Scatter Plot
        st.subheader("Customer Segments (PCA Visualization)")
        hover_cols = numeric_cols + cat_cols + ['Cluster']
        fig_pca = px.scatter(
            df_results, 
            x='PC1', 
            y='PC2', 
            color='Cluster',
            title=f"2D PCA Visualization of {k_slider} Clusters",
            hover_data=[col for col in hover_cols if col in df_results.columns],
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig_pca, use_container_width=True) # Note: Keeping use_container_width as 'width' is not a direct replacement for plotly
        
        # Show Cluster Profiles
        st.subheader("Cluster Profiles (Mean Values)")
        st.markdown("This table shows the average values for each numeric feature by cluster.")
        cluster_profile = df_results.groupby('Cluster')[numeric_cols].mean().reset_index()
        st.dataframe(cluster_profile, width='stretch')

    # --- Tab 4: Predict New Customer ---
    with tab4:
        st.header("Predict New Customer Segment")
        st.markdown("Enter the details of a new customer to predict their segment.")
        
        # Use a form for input
        with st.form(key="prediction_form"):
            input_data = {}
            
            # Create input fields dynamically
            for col in numeric_cols:
                input_data[col] = st.number_input(
                    f"Enter {col}", 
                    value=float(df_analysis[col].median())
                )
            
            for col in cat_cols:
                unique_vals = df_analysis[col].unique().tolist()
                input_data[col] = st.selectbox(
                    f"Select {col}", 
                    options=unique_vals,
                    index=0
                )
            
            submit_button = st.form_submit_button(label="Predict Segment")

        if submit_button:
            # Create a DataFrame from the input
            new_customer_df = pd.DataFrame([input_data])
            
            # Preprocess the new data
            try:
                new_data_processed = preprocessor.transform(new_customer_df)
                
                # Predict the cluster
                prediction = kmeans_model.predict(new_data_processed)
                
                st.success(f"This customer belongs to *Cluster {prediction[0]}*")
                
                # Show profile of the predicted cluster
                st.subheader(f"Profile of Predicted Cluster {prediction[0]}")
                st.dataframe(cluster_profile[cluster_profile['Cluster'] == str(prediction[0])], width='stretch')
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

except Exception as e:
    st.error(f"An error occurred during data processing: {e}")
    st.error("Please check your uploaded CSV file. It may have an incorrect format, missing columns, or data types that don't match the expected schema (numeric and categorical).")