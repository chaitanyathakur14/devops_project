import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Online Retail MLOps & Customer Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# Custom Styling for Clean Presentation
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("models/retail_model.joblib")

# Load Processed Dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_retail.csv")

model = load_model()
df = load_data()

# Ensure datetime format for calculations
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# App Title & Overview
st.title("Online Retail Customer Value Platform")
st.markdown("An end-to-end data engineering, version-controlled machine learning, and interactive analytics project.")

st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Project Architecture & Flow", 
    "Live Model Inference", 
    "Exploratory Data Analysis", 
    "Model Performance & Metrics"
])

# =========================================================================
# TAB 1: ARCHITECTURE & PROJECT FLOW
# =========================================================================
with tab1:
    st.subheader("System Architecture & MLOps Pipeline Flow")
    st.markdown("This project follows industry-standard MLOps practices to decouple heavy data storage from version control while automating reproducible training and deployment.")

    col1, col2 = st.columns([1.1, 0.9], gap="large")
    
    with col1:
        st.markdown("### **Pipeline Components**")
        st.markdown("""
        * **1. Data Ingestion & Version Control (DVC):** Raw transactional data (`online_retail_II.xlsx`) spanning two years is version-controlled via **DVC**, keeping heavy binaries out of Git history.
        * **2. Data Engineering (`src/process_data.py`):** Cleans missing identifiers, filters out invoice cancellations, and computes transaction totals (`Quantity * Price`).
        * **3. Feature Engineering & Modeling (`src/train.py`):** Aggregates logs into **RFM (Recency, Frequency, Monetary)** features and trains a robust **Random Forest Regressor**.
        * **4. Artifact Management:** Serializes trained weights into `models/retail_model.joblib`.
        * **5. Interactive UI (`app.py`):** Serves real-time predictions and visualizations via Streamlit.
        """)
        
    with col2:
        st.markdown("### **Workflow Schematic**")
        st.info(" **Raw Data** \n`data/raw/online_retail_II.xlsx`\n(Managed via DVC)")
        st.markdown(" **Processing (`process_data.py`)**")
        st.success(" **Cleaned Dataset** \n`data/processed/cleaned_retail.csv`")
        st.markdown(" **Training (`train.py`)**")
        st.warning("**Trained Model Artifact** \n`models/retail_model.joblib`")
        st.markdown(" **Deployment (`app.py`)**")
        st.error(" **Streamlit Web Interface**")

# =========================================================================
# TAB 2: LIVE MODEL INFERENCE
# =========================================================================
with tab2:
    st.subheader("Customer Lifetime Value (CLV) Prediction")
    st.markdown("Simulate a customer profile by adjusting their RFM characteristics to predict total expected monetary expenditure.")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.slider(
            "Recency (Days Since Last Purchase)", 
            min_value=0, max_value=365, value=30,
            help="Fewer days indicate a more recently active customer."
        )
    with col2:
        frequency = st.slider(
            "Frequency (Unique Orders Placed)", 
            min_value=1, max_value=100, value=5,
            help="Total number of distinct purchase transactions."
        )
    with col3:
        avg_order_value = st.number_input(
            "Average Order Value ($)", 
            min_value=1.0, max_value=5000.0, value=150.0,
            help="Mean amount spent per transaction invoice."
        )

    st.markdown("")

    if st.button("Generate Prediction", type="primary", use_container_width=True):
        input_data = np.array([[recency, frequency, avg_order_value]])
        prediction = model.predict(input_data)[0]

        res_col1, res_col2 = st.columns([1, 1], gap="large")

        with res_col1:
            st.metric(
                label="Predicted Total Customer Spend",
                value=f"${prediction:,.2f}"
            )
            st.markdown(f"""
            **Evaluated Parameters:**
            - Inactive Duration: **{recency} days**
            - Transaction Count: **{frequency} orders**
            - Basket Size Average: **${avg_order_value:,.2f}**
            """)

        with res_col2:
            st.markdown("#### Feature Comparison vs Dataset Medians")
            cust_agg = df.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                'Invoice': 'nunique',
                'TotalSpend': 'sum'
            })
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            metrics_labels = ['Recency', 'Frequency', 'Avg Order Value']
            user_vals = [recency, frequency, avg_order_value]
            median_vals = [cust_agg['InvoiceDate'].median(), cust_agg['Invoice'].median(), df['TotalSpend'].mean() / max(frequency, 1)]
            
            x = np.arange(len(metrics_labels))
            width = 0.35
            
            ax.bar(x - width/2, user_vals, width, label='Simulated User', color='#1f77b4')
            ax.bar(x + width/2, median_vals, width, label='Dataset Median', color='#aec7e8')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_labels, fontsize=9)
            ax.legend(fontsize=9, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

# =========================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS
# =========================================================================
with tab3:
    st.subheader("Transactional Data Exploratory Visualizations")
    st.markdown("Advanced analytical charts generated directly from the processed retail dataset.")

    cust_summary = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalSpend': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Monetary': 'TotalSpend'})

    eda_option = st.selectbox(
        "Select Analytical View", 
        ["Customer Spend Distribution (Log Scale)", "Recency vs. Total Spend Correlation", "Top 10 Countries by Revenue"]
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))

    if eda_option == "Customer Spend Distribution (Log Scale)":
        sns.histplot(cust_summary['TotalSpend'], bins=50, kde=True, color='teal', ax=ax)
        ax.set_title("Distribution of Customer Total Spend", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Total Spend ($)", fontsize=10)
        ax.set_ylabel("Customer Count", fontsize=10)
        ax.set_xlim(0, cust_summary['TotalSpend'].quantile(0.95))

    elif eda_option == "Recency vs. Total Spend Correlation":
        sample_df = cust_summary.sample(min(2000, len(cust_summary)))
        sns.scatterplot(data=sample_df, x='Recency', y='TotalSpend', alpha=0.5, color='darkorange', ax=ax)
        ax.set_title("Customer Recency vs. Total Spend", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Recency (Days Since Last Order)", fontsize=10)
        ax.set_ylabel("Total Spend ($)", fontsize=10)
        ax.set_ylim(0, sample_df['TotalSpend'].quantile(0.95))

    else:
        country_spend = df.groupby('Country')['TotalSpend'].sum().sort_values(ascending=False).head(10)
        sns.barplot(x=country_spend.values, y=country_spend.index, palette='viridis', ax=ax)
        ax.set_title("Top 10 Countries by Total Revenue", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Total Revenue ($)", fontsize=10)
        ax.set_ylabel("Country", fontsize=10)

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("#### Summary Statistical Overview")
    st.dataframe(cust_summary.describe(), use_container_width=True)

# =========================================================================
# TAB 4: MODEL PERFORMANCE
# =========================================================================
with tab4:
    st.subheader("Model Evaluation & Regression Diagnostics")
    st.markdown("Performance metrics and validation insights for the trained Random Forest Regressor.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        ### **Evaluation Metrics**
        * **Algorithm:** Random Forest Regressor (`n_estimators=100`)
        * **Train/Test Split:** 80% Training, 20% Testing
        * **Target Variable:** Total Customer Monetary Spend ($)
        
        | Metric | Score / Value | Interpretation |
        | :--- | :--- | :--- |
        | **$R^2$ Score** | **0.58** | Explains 58% of variance in customer spend. |
        | **RMSE** | **$12,439.67** | Standard error deviation on revenue data. |
        """)

    with col2:
        st.markdown("### **Feature Importance Analysis**")
        feature_importance = model.feature_importances_
        features = ['Recency', 'Frequency', 'Avg Order Value']
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(features, feature_importance, color='#2ca02c')
        ax.set_title("Random Forest Feature Importances", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Importance Weight", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>End-to-End MLOps Project | Built with Git, DVC, Scikit-Learn, and Streamlit</p>", unsafe_allow_html=True)