import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px

st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")