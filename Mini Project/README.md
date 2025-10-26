# 🛍️ Customer Segmentation for Retail Using K-Means and PCA

Customer segmentation is one of the most powerful strategies in modern retail analytics, enabling businesses to understand, categorize, and target customers more effectively. This project, **Customer Segmentation for Retail Using K-Means and PCA**, leverages unsupervised machine learning to identify meaningful customer groups based on their purchasing patterns and demographic characteristics. The goal is to help retailers design personalized marketing strategies, optimize promotions, and enhance overall customer satisfaction.

This system uses the **K-Means clustering algorithm** to identify natural groupings among customers and employs **Principal Component Analysis (PCA)** for dimensionality reduction and visualization. The preprocessing pipeline handles missing data, encodes categorical variables, and standardizes numeric features before clustering. The **Elbow Method** and **Silhouette Score** are applied to determine the optimal number of clusters, ensuring that the resulting groups are both interpretable and statistically sound.

An interactive **Streamlit** web application is built to provide a simple and intuitive interface for analysis. Users can upload their own retail dataset, explore data insights, visualize cluster formation in real-time, and even predict the cluster assignment of a new customer. The dashboard is divided into four main sections: *Data Overview*, *Optimal K Analysis*, *Segmentation Results*, and *Predict New Customer*. The first section displays dataset statistics and detected feature types. The second helps determine the ideal cluster count using the Elbow and Silhouette methods. The third provides an interactive PCA scatter plot to visualize segment distribution, along with mean cluster profiles. Finally, the fourth section allows users to input new customer details and instantly predict their likely segment.

The app is implemented in **Python**, using libraries such as `pandas`, `numpy`, `scikit-learn`, and `plotly` for processing, modeling, and visualization. Caching via `st.cache_data` and `st.cache_resource` ensures fast performance even on large datasets. The project is designed with modular, readable code for easy maintenance and scalability. It demonstrates how unsupervised learning can turn raw customer data into actionable business intelligence, supporting better decision-making and personalized retail experiences.

To run this project locally, install the dependencies listed in `requirements.txt` and launch the Streamlit app with the command:

```bash
streamlit run app.py
