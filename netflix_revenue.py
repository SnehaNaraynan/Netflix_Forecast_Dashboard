import streamlit as st
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import boto3
import json

st.title("Netflix Revenue Caster")

# Upload or use default CSV
upload = st.file_uploader("Upload your Revenue data", type="csv")
if upload:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv("/Users/snehanarayanan/Documents/Netflix Project/netflix_revenue_updated_.csv") 


target = st.selectbox("What do you want to forecast?", ["UCAN Revenue", "UCAN ARPU"], key="forecast_target")


if target == "UCAN Revenue":
    df_model = df[['Date', 'UCAN Streaming Revenue']].dropna()
    ylabel = "Revenue"
else:
    df_model = df[['Date', 'UCAN ARPU']].dropna()
    ylabel = "ARPU"


if target == "UCAN Revenue":
    df_model = df[['Date', 'UCAN Streaming Revenue']].dropna()
    ylabel = "Revenue"
else:
    df_model = df[['Date', 'UCAN ARPU']].dropna()
    ylabel = "ARPU"

# Forecasting model
model_type = st.radio("Choose Model", ["Prophet", "ARIMA"])

df_model['Date'] = pd.to_datetime(df_model['Date'])
df_model = df_model.rename(columns={"Date": "ds", df_model.columns[1]: "y"})

steps = st.slider("Forecast Quarters Ahead", 1, 6, 4)


# Build & forecast
if model_type == "Prophet":
    m = Prophet()
    m.fit(df_model)
    future = m.make_future_dataframe(periods=steps, freq="Q")
    forecast = m.predict(future)
    
    # Plot
    fig = m.plot(forecast)
    fig.gca().set_xlabel("Quarter")
    fig.gca().set_ylabel(target)
    st.pyplot(fig)

    # Format result
    result = forecast.tail(steps)[['ds', 'yhat']]
    result = result.rename(columns={'ds': 'Quarter', 'yhat': f"Forecasted {target}"})
    result[f"Forecasted {target}"] = result[f"Forecasted {target}"].apply(lambda x: f"${x / 1e9:.2f}B")

else:
    df_arima = df_model.set_index("ds")
    arima_model = ARIMA(df_arima['y'], order=(1, 1, 1)).fit()
    forecast = arima_model.forecast(steps=steps)
    forecast.index = pd.date_range(start=df_arima.index[-1] + pd.offsets.QuarterEnd(), periods=steps, freq='Q')
    
    # Plot
    fig, ax = plt.subplots()
    df_arima['y'].plot(ax=ax, label='Historical')
    forecast.plot(ax=ax, label='Forecast')
    ax.set_xlabel("Quarter")
    ax.set_ylabel(target)
    plt.legend()
    st.pyplot(fig)

    # Format result
    result = forecast.reset_index()
    result.columns = ['Quarter', f"Forecasted {target}"]
    result[f"Forecasted {target}"] = result[f"Forecasted {target}"].apply(lambda x: f"${x / 1e9:.2f}B")

st.subheader("Forecast Summary")
st.dataframe(result)


# Initialize Bedrock client

from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env automatically

aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")

import boto3

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=region,
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret
)
print(f"KEY: {aws_key}")
print(f"REGION: {region}")


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


if st.button("Generate Claude Insights"):
    prompt = f"""Netflix's {target} forecast is:\n{result.to_string(index=False)}\n\nBased on this, give 3 business recommendations to improve revenue or ARPU in the UCAN region. Be concise and actionable."""
    insights = get_claude_recommendations(prompt)
    st.subheader("Claude AI Recommendations")
    st.write(insights)
