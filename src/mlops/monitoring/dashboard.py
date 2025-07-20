import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os

st.title("Prediction Drift Monitoring")

# Database connection
DATABASE_URI = os.getenv(
    "METRICS_DB_URI", "postgresql://user:pass@localhost:5432/prediction_metrics"
)
engine = create_engine(DATABASE_URI)

# Load metrics
df = pd.read_sql("SELECT * FROM prediction_metrics ORDER BY timestamp DESC", engine)

# Display charts
st.subheader("Drifted Columns Over Time")
st.line_chart(df.set_index("timestamp")["drifted_columns"])

st.subheader("Missing Value Share")
st.line_chart(df.set_index("timestamp")["missing_value_share"])

# Show table
st.subheader("Raw Metrics Data")
st.dataframe(df)
