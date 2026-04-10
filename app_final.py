import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px

st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password", 
    help="You can find your API key at https://platform.openai.com/account/api-keys"
)

def classify_sentiment_openai(review_text):
    """
    Classify the sentiment of a customer review using OpenAI's GPT-4o model.
    Parameters:
        review_text (str): The customer review text to be classified.
    Returns:
        str: The sentiment classification of the review as a single word, "positive", "negative", or "neutral".
    """
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content


# Example CSV format
with st.expander("📋 See example CSV format"):
    st.markdown("Your CSV file should have at least one text column containing the reviews. Here's an example:")
    example_df = pd.DataFrame({
        "Reviews": [
            "The food was absolutely delicious, we'll be back!",
            "Slow service, the wait was not worth it.",
            "Decent meal, nothing too special but okay overall."
        ]
    })
    st.dataframe(example_df, use_container_width=True)
    st.caption("The column name can be anything — you'll be able to select it after uploading.")

# CSV file uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with restaurant reviews", 
    type=["csv"])

# Initialise session state
if "reviews_df" not in st.session_state:
    st.session_state.reviews_df = None
if "sentiment_counts" not in st.session_state:
    st.session_state.sentiment_counts = None

# Show column selector if file is uploaded
review_column = None
if uploaded_file is not None:
    preview_df = pd.read_csv(uploaded_file)
    text_columns = preview_df.select_dtypes(include="object").columns
    if len(text_columns) == 0:
        st.error("No text columns found in the uploaded file.")
    else:
        review_column = st.selectbox(
            "Select the column with the customer reviews",
            text_columns
        )

# Generate and Reset buttons
st.markdown("---")
col_gen, col_reset = st.columns([3, 1])
with col_gen:
    generate_clicked = st.button("🚀 Analyse Sentiment", use_container_width=True)
with col_reset:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.session_state.reviews_df = None
    st.session_state.sentiment_counts = None
    st.rerun()

if generate_clicked:
    # Validate all inputs
    errors = []
    if not openai_api_key:
        errors.append("🔑 OpenAI API key is missing. Please enter it in the sidebar.")
    if uploaded_file is None:
        errors.append("📄 CSV file is missing. Please upload a file.")
    if uploaded_file is not None and review_column is None:
        errors.append("📋 No valid text column found in the uploaded file.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        reviews_df = pd.read_csv(uploaded_file)
        with st.spinner("Analysing sentiments..."):
            reviews_df["sentiment"] = reviews_df[review_column].apply(classify_sentiment_openai)
        reviews_df["sentiment"] = reviews_df["sentiment"].str.strip().str.strip(".").str.title()
        st.session_state.reviews_df = reviews_df
        st.session_state.sentiment_counts = reviews_df["sentiment"].value_counts()

# Display results if they exist in session state
if st.session_state.reviews_df is not None:
    reviews_df = st.session_state.reviews_df
    sentiment_counts = st.session_state.sentiment_counts

    # Create 3 columns to display the 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        positive_count = sentiment_counts.get("Positive", 0)
        st.metric("Positive",
                  positive_count,
                  f"{positive_count / len(reviews_df) * 100:.2f}%")

    with col2:
        neutral_count = sentiment_counts.get("Neutral", 0)
        st.metric("Neutral",
                  neutral_count,
                  f"{neutral_count / len(reviews_df) * 100:.2f}%")

    with col3:
        negative_count = sentiment_counts.get("Negative", 0)
        st.metric("Negative",
                  negative_count,
                  f"{negative_count / len(reviews_df) * 100:.2f}%")

    # Display pie chart
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution'
    )
    st.plotly_chart(fig)

    # Download button for the results
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=reviews_df.to_csv(index=False).encode("utf-8"),
        file_name="reviews_with_sentiment.csv",
        mime="text/csv"
    )
