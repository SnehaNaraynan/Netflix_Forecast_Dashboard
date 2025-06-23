import streamlit as st
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import boto3
import json
from dotenv import load_dotenv
import os

# === Custom Styling ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    h1, h3 { color: #2E86AB; }
    .stDataFrame { font-size: 16px; }
    .stRadio > div { flex-direction: row; }
    </style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown("<h1 style='font-size: 38px;'>Netflix Revenue Caster</h1>", unsafe_allow_html=True)

# === Sidebar Controls ===
with st.sidebar:
    st.markdown("## Upload & Settings")
    upload = st.file_uploader("Upload your Revenue data", type="csv")
    target = st.selectbox("What do you want to forecast?", ["UCAN Revenue", "UCAN ARPU"], key="forecast_target")
    model_type = st.radio("Choose Model", ["Prophet", "ARIMA"], key="model_choice")
    steps = st.slider("Forecast Quarters Ahead", 1, 6, 4)

# === Load Data ===
if upload:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv("/Users/snehanarayanan/Documents/Netflix Project/netflix_revenue_updated_.csv")

if target == "UCAN Revenue":
    df_model = df[['Date', 'UCAN Streaming Revenue']].dropna()
    ylabel = "Revenue"
else:
    df_model = df[['Date', 'UCAN ARPU']].dropna()
    ylabel = "ARPU"

df_model['Date'] = pd.to_datetime(df_model['Date'])
df_model = df_model.rename(columns={"Date": "ds", df_model.columns[1]: "y"})

# === Forecasting ===
if model_type == "Prophet":
    m = Prophet()
    m.fit(df_model)
    future = m.make_future_dataframe(periods=steps, freq="Q")
    forecast = m.predict(future)

    fig = m.plot(forecast)
    fig.gca().set_xlabel("Quarter")
    fig.gca().set_ylabel(target)

    result = forecast.tail(steps)[['ds', 'yhat']]
    result = result.rename(columns={'ds': 'Quarter', 'yhat': f"Forecasted {target}"})
    if "Revenue" in target:
        result[f"Forecasted {target}"] = result[f"Forecasted {target}"].apply(lambda x: f"${x / 1e9:.2f}B")
    else:  # ARPU or similar metric
        result[f"Forecasted {target}"] = result[f"Forecasted {target}"].apply(lambda x: f"${x:.2f}")


else:
    df_arima = df_model.set_index("ds")
    arima_model = ARIMA(df_arima['y'], order=(1, 1, 1)).fit()
    forecast = arima_model.forecast(steps=steps)
    forecast.index = pd.date_range(start=df_arima.index[-1] + pd.offsets.QuarterEnd(), periods=steps, freq='Q')

    fig, ax = plt.subplots()
    df_arima['y'].plot(ax=ax, label='Historical')
    forecast.plot(ax=ax, label='Forecast')
    ax.set_xlabel("Quarter")
    ax.set_ylabel(target)
    ax.legend()

    result = forecast.reset_index()
    result.columns = ['Quarter', f"Forecasted {target}"]
    if "Revenue" in target:
        result[f"Forecasted {target}"] = result[f"Forecasted {target}"].apply(lambda x: f"${x / 1e9:.2f}B")
    else:  # ARPU or similar metric
        result[f"Forecasted {target}"] = result[f"Forecasted {target}"].apply(lambda x: f"${x:.2f}")

# === Display Forecast: Chart + Table ===
st.markdown("### Forecast Visualization")

col1, col2 = st.columns([2, 1])

with col1:
    st.pyplot(fig)

with col2:
    st.markdown("### Forecast Summary")
    st.dataframe(result, use_container_width=True)

# === Claude AI Section ===

load_dotenv()

aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")

if all([aws_key, aws_secret, region]):
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region,
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
            f"Netflix's {target} forecast for the next {steps} quarters is:\n"
            f"{result.to_string(index=False)}\n\n"
            f"Give 3 actionable recommendations to improve this metric in the UCAN region."
        )
        try:
            insights = get_claude_recommendations(prompt)
            st.write(insights)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("Claude AI not configured — missing AWS credentials.")