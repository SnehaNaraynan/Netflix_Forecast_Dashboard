import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import boto3, json, os
from dotenv import load_dotenv

# === Streamlit Layout & Style ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .stApp { font-family: 'Segoe UI', sans-serif; font-size: 18px; }
        h1, h3, h4 { color: #E50914; }
        .stDataFrame { font-size: 16px; }
        .stMetric { background-color: #e5e5e5; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 42px;'>🎬 Netflix Regional Performance Dashboard</h1>", unsafe_allow_html=True)

# === Sidebar Inputs ===
with st.sidebar:
    st.markdown("## 📂 Upload & Settings")
    upload = st.file_uploader("Upload your Netflix data", type="csv")
    region_display_names = {
    "US & Canada": "UCAN",
    "Europe, Middle East & Africa": "EMEA",
    "Latin America": "LATAM",
    "Asia Pacific": "APAC"
    }
    region_label = st.selectbox("Region", list(region_display_names.keys()), key="region")
    region = region_display_names[region_label]
    steps = st.slider("Forecast Quarters Ahead", 1, 6, 4)

# === Load Data ===
if upload:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv("/Users/snehanarayanan/Documents/Netflix Project/netflix_revenue_updated_.csv")

# === Region Columns
rev_col = f"{region} Streaming Revenue"
arpu_col = f"{region} ARPU"
members_col = f"{region} Members"

# === Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df_rev = df[['Date', rev_col]].dropna().rename(columns={"Date": "ds", rev_col: "y"})
df_arpu = df[['Date', arpu_col]].dropna().rename(columns={"Date": "ds", arpu_col: "y"})

# === Forecast Function
def forecast_series(df_model, steps):
    prophet_model = Prophet()
    prophet_model.fit(df_model)
    future = prophet_model.make_future_dataframe(periods=steps, freq='Q')
    forecast = prophet_model.predict(future)
    return forecast, prophet_model

forecast_rev, model_rev = forecast_series(df_rev, steps)
forecast_arpu, model_arpu = forecast_series(df_arpu, steps)

rev_result = forecast_rev.tail(steps)[['ds', 'yhat']].rename(columns={'ds': 'Quarter', 'yhat': f"{region} Revenue Forecast"})
arpu_result = forecast_arpu.tail(steps)[['ds', 'yhat']].rename(columns={'ds': 'Quarter', 'yhat': f"{region} ARPU Forecast"})

# === Format forecasted values
rev_result[f"{region} Revenue Forecast"] = rev_result[f"{region} Revenue Forecast"].apply(lambda x: f"${x / 1e9:.2f}B")
arpu_result[f"{region} ARPU Forecast"] = arpu_result[f"{region} ARPU Forecast"].apply(lambda x: f"${x:.2f}")

# === Section: KPI Snapshot ===
latest_rev = df_rev['y'].iloc[-1]
latest_arpu = df_arpu['y'].iloc[-1]
latest_members = df[members_col].iloc[-1]

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric(label=f"{region} Latest Revenue", value=f"${latest_rev / 1e9:.2f}B")
col_kpi2.metric(label=f"{region} Latest ARPU", value=f"${latest_arpu:.2f}")
col_kpi3.metric(label=f"{region} Paid Members", value=f"{int(latest_members):,}")

# === Section: Revenue & ARPU Charts
st.markdown("### Forecast Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig_rev = model_rev.plot(forecast_rev)
    fig_rev.gca().set_xlabel("Quarter")
    fig_rev.gca().set_ylabel("Revenue")
    st.pyplot(fig_rev)

with col2:
    fig_arpu = model_arpu.plot(forecast_arpu)
    fig_arpu.gca().set_xlabel("Quarter")
    fig_arpu.gca().set_ylabel("ARPU")
    st.pyplot(fig_arpu)

# === Section: Forecast Tables
st.markdown("### Forecast Tables")

col_table1, col_table2 = st.columns(2)
with col_table1:
    st.dataframe(rev_result, use_container_width=True)
with col_table2:
    st.dataframe(arpu_result, use_container_width=True)

# === Claude AI Recommendations ===
load_dotenv()
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
region_aws = os.getenv("AWS_DEFAULT_REGION")

if all([aws_key, aws_secret, region_aws]):
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region_aws,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret
    )

    def get_claude_recommendations(prompt):
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 300,
            "temperature": 0.7,
        })
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        response_body = json.loads(response['body'].read())
        return response_body.get("completion")

    st.markdown("### Claude AI Recommendations")
    if st.button("Generate AI Insights"):
        prompt = (
            f"Netflix's forecast for the {region} region over the next {steps} quarters is:\n\n"
            f"{rev_result.to_string(index=False)}\n\n"
            f"{arpu_result.to_string(index=False)}\n\n"
            f"Based on these forecasts, suggest 3 realistic strategies Netflix could use to grow revenue and improve ARPU in {region}."
        )
        try:
            insights = get_claude_recommendations(prompt)
            st.write(insights)
        except Exception as e:
            st.error(f"Claude Error: {e}")
else:
    st.warning("Claude AI not configured — check AWS credentials.")
