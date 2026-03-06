import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Assistant")
st.write("Upload a dataset to automatically perform exploratory data analysis and generate insights.")

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# -------------------------------
# LOAD DATA
# -------------------------------

if uploaded_file:

    df = pd.read_csv(uploaded_file)

# -------------------------------
# DATASET PREVIEW
# -------------------------------

    st.header("Dataset Preview")
    st.dataframe(df.head())

# -------------------------------
# KPI DASHBOARD
# -------------------------------

    st.header("Dataset Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

# -------------------------------
# DATA QUALITY REPORT
# -------------------------------

    st.header("Data Quality Report")

    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    st.write("Total Missing Values:", missing_values)
    st.write("Duplicate Rows:", duplicate_rows)

# -------------------------------
# MISSING VALUE ANALYSIS
# -------------------------------

    st.header("Missing Value Summary")

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Values": missing,
        "Percentage (%)": missing_percent
    })

    st.dataframe(missing_df[missing_df["Missing Values"] > 0])

# -------------------------------
# COLUMN TYPE DETECTION
# -------------------------------

    st.header("Column Type Detection")

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    col1, col2 = st.columns(2)

    with col1:
        st.write("Numeric Columns")
        st.write(list(numeric_cols))

    with col2:
        st.write("Categorical Columns")
        st.write(list(categorical_cols))

# -------------------------------
# SUMMARY STATISTICS
# -------------------------------

    st.header("Summary Statistics")

    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())

# -------------------------------
# HISTOGRAM VISUALIZATION
# -------------------------------

    if len(numeric_cols) > 0:

        st.header("Histogram Visualization")

        selected_col = st.selectbox("Select numeric column", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[selected_col].dropna(), kde=True)

        st.pyplot(fig)

# -------------------------------
# AUTOMATIC CHART GENERATOR
# -------------------------------

    if len(numeric_cols) > 0:

        st.header("Automatic Chart Generator")

        for col in numeric_cols:

            st.write(f"Distribution of {col}")

            fig, ax = plt.subplots()

            sns.histplot(df[col].dropna(), kde=True)

            st.pyplot(fig)

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------

    if len(numeric_cols) > 1:

        st.header("Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig2, ax2 = plt.subplots()

        sns.heatmap(corr, annot=True, cmap="coolwarm")

        st.pyplot(fig2)

# -------------------------------
# STRONG CORRELATION INSIGHTS
# -------------------------------

    if len(numeric_cols) > 1:

        st.header("Strong Correlation Insights")

        corr_matrix = df[numeric_cols].corr()

        for col in corr_matrix.columns:
            for row in corr_matrix.index:

                if abs(corr_matrix.loc[row, col]) > 0.7 and row != col:

                    st.write(f"{row} ↔ {col} : {corr_matrix.loc[row, col]:.2f}")

# -------------------------------
# CATEGORICAL ANALYSIS
# -------------------------------

    if len(categorical_cols) > 0:

        st.header("Categorical Value Analysis")

        selected_cat = st.selectbox("Select categorical column", categorical_cols)

        st.dataframe(df[selected_cat].value_counts().head(10))

# -------------------------------
# OUTLIER DETECTION
# -------------------------------

    if len(numeric_cols) > 0:

        st.header("Outlier Detection (IQR Method)")

        selected_outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)

        Q1 = df[selected_outlier_col].quantile(0.25)
        Q3 = df[selected_outlier_col].quantile(0.75)

        IQR = Q3 - Q1

        outliers = df[
            (df[selected_outlier_col] < Q1 - 1.5 * IQR) |
            (df[selected_outlier_col] > Q3 + 1.5 * IQR)
        ]

        st.write("Number of Outliers:", len(outliers))

# -------------------------------
# DATA CLEANING
# -------------------------------

    st.header("Clean Dataset")

    clean_df = df.copy()

    clean_df = clean_df.drop_duplicates()

    for col in clean_df.select_dtypes(include=['number']).columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    for col in clean_df.select_dtypes(include=['object']).columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])

    st.write("Preview of Cleaned Dataset")
    st.dataframe(clean_df.head())

# -------------------------------
# DOWNLOAD CLEAN DATA
# -------------------------------

    csv = clean_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

# -------------------------------
# DOWNLOAD ANALYSIS REPORT
# -------------------------------

    st.header("Download Dataset Report")

    report = f"""
DATASET ANALYSIS REPORT

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Missing Values: {df.isnull().sum().sum()}
Duplicate Rows: {df.duplicated().sum()}

Numeric Columns: {list(numeric_cols)}
Categorical Columns: {list(categorical_cols)}
"""

    st.download_button(
        label="Download Dataset Report",
        data=report,
        file_name="dataset_report.txt",
        mime="text/plain"
    )