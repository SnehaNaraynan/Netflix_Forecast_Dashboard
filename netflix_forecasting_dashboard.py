import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import boto3, json, os
from dotenv import load_dotenv

# === Page Setup ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    h1, h3 { color: #2E86AB; }
    .stDataFrame { font-size: 16px; }
    .stRadio > div { flex-direction: row; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 38px;'>🎬 Netflix Regional Dashboard</h1>", unsafe_allow_html=True)

# === Sidebar Inputs ===
with st.sidebar:
    st.markdown("## 📂 Upload & Settings")
    upload = st.file_uploader("Upload your Netflix data", type="csv")
    region = st.selectbox("Region", ["UCAN", "EMEA", "LATAM", "APAC"], key="region")
    metric_type = st.radio("Metric", ["Revenue", "ARPU"], key="metric")
    steps = st.slider("Forecast Quarters Ahead", 1, 6, 4)

# === Load Data ===
if upload:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv("/Users/snehanarayanan/Documents/Netflix Project/netflix_revenue_updated_.csv")

# Column Mappings
region_options = {
    "Revenue": {
        "UCAN": "UCAN Streaming Revenue",
        "EMEA": "EMEA Streaming Revenue",
        "LATAM": "LATM Streaming Revenue",
        "APAC": "APAC Streaming Revenue"
    },
    "ARPU": {
        "UCAN": "UCAN ARPU",
        "EMEA": "EMEA ARPU",
        "LATAM": "LATM ARPU",
        "APAC": "APAC ARPU"
    }
}
metric_col = region_options[metric_type][region]
df_model = df[['Date', metric_col]].dropna()
df_model['Date'] = pd.to_datetime(df_model['Date'])
df_model = df_model.rename(columns={"Date": "ds", metric_col: "y"})

# === Forecasting Function ===
def run_prophet(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='Q')
    forecast = model.predict(future)
    y_true = df['y'][-4:]
    y_pred = forecast.loc[forecast['ds'].isin(df['ds'].tail(4)), 'yhat']
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return forecast, mape, model

def run_arima(df):
    df_arima = df.set_index("ds")
    model = ARIMA(df_arima['y'], order=(1,1,1)).fit()
    forecast = model.forecast(steps=steps)
    forecast.index = pd.date_range(start=df_arima.index[-1] + pd.offsets.QuarterEnd(), periods=steps, freq='Q')
    y_true = df_arima['y'][-4:]
    y_pred = model.fittedvalues[-4:]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return forecast, mape, df_arima, model

# === Auto Model Selection ===
forecast, selected_model, fig, result = None, None, None, None

with st.spinner("Training models and comparing performance..."):
    prophet_forecast, mape_prophet, prophet_model = run_prophet(df_model)
    arima_forecast, mape_arima, df_arima, arima_model = run_arima(df_model)

    if mape_prophet < mape_arima:
        selected_model = "Prophet"
        forecast = prophet_forecast.tail(steps)[['ds', 'yhat']]
        forecast = forecast.rename(columns={"ds": "Quarter", "yhat": f"Forecasted {region} {metric_type}"})
        fig = prophet_model.plot(prophet_forecast)
        fig.gca().set_xlabel("Quarter")
        fig.gca().set_ylabel(f"{region} {metric_type}")
    else:
        selected_model = "ARIMA"
        forecast = arima_forecast.reset_index()
        forecast.columns = ["Quarter", f"Forecasted {region} {metric_type}"]
        fig, ax = plt.subplots()
        df_arima['y'].plot(ax=ax, label='Historical')
        arima_forecast.plot(ax=ax, label='Forecast')
        ax.set_xlabel("Quarter")
        ax.set_ylabel(f"{region} {metric_type}")
        ax.legend()

# === Format Output Table ===
if "Revenue" in metric_type:
    forecast[f"Forecasted {region} {metric_type}"] = forecast[f"Forecasted {region} {metric_type}"].apply(lambda x: f"${x / 1e9:.2f}B")
else:
    forecast[f"Forecasted {region} {metric_type}"] = forecast[f"Forecasted {region} {metric_type}"].apply(lambda x: f"${x:.2f}")

# === Display ===
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### Forecast Visualization ({selected_model})")
    st.pyplot(fig)

with col2:
    st.markdown("### Forecast Summary")
    st.dataframe(forecast, use_container_width=True)

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
            f"Netflix's {region} {metric_type} forecast for the next {steps} quarters is:\n"
            f"{forecast.to_string(index=False)}\n\n"
            f"Give 3 specific, realistic recommendations to improve this metric in the {region} region."
        )
        try:
            insights = get_claude_recommendations(prompt)
            st.write(insights)
        except Exception as e:
            st.error(f"Claude Error: {e}")
else:
    st.warning("Claude AI not configured — check AWS credentials.")
