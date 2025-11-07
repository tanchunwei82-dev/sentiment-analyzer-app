import streamlit as st
import langchain_ollama
import pandas as pd
import plotly

from langchain_ollama.llms import OllamaLLM
# OpenAI API Key Input
openai_api_key = st.sidebar.text_input(
    "Enter your OpoeAI API Key",
    type="password",
    help="You can find your API key at http://platform.openai.com/account/api"
)

def classify_sentiment_ollama(review_text):

    # Load the model
    #llm = OllamaLLM(model="llama3.2")  # Adjust the model name as needed, for example "deepseek-r1:7b" or "mistral"
    llm = OllamaLLM(model="mistral")

    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    result = llm.invoke(prompt)

    return result[1:]

st.title("🍪 Customer Review Sentiment Analyzer")
# st.title("_Streamlit_ is :blue[cool] :sunglasses:")

st.markdown("This anlayzes the sentiment of cutomer reviews to gain insights into their opinions")

# user_input = st.text_input("Enter a customer review", "")
# st.write("The current user review is:", user_input)

# import pandas as pd
# df = pd.read_csv("reviews.csv")
# st.write(df)

# CSV file uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with restaurant reviews",
    type=["csv"]
)

# Once the user uploads a csv file:
if uploaded_file is not None:
    # Read the file
    reviews_df = pd.read_csv(uploaded_file)

    #Check if the data has a text column
    text_columns = reviews_df.select_dtypes(include="object").columns

    if len(text_columns)==0:
        st.error("No text column!")

    # Show a dropdown menu to select the review column
    review_column = st.selectbox(
        "Select the column with the customer reviews",
        text_columns
    )

    # Analyze the sentiment of the selected colum
    reviews_df["Sentiment"] = reviews_df[review_column].apply(classify_sentiment_ollama)
    
    # Display the sentiment distribution in metrics in 3 columns: Positive, Negative, Neutral
    # Make the strings in the sentiment column title
    reviews_df["Sentiment"] = reviews_df["Sentiment"].str.title()
    sentiment_counts = reviews_df["Sentiment"].value_counts()
    # st.write(reviews_df)
    # st.write(sentiment_counts)

    # Create tree columsn to dissplay the 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        # Show the number of positives rvviews and the percentage
        positive_count = sentiment_counts.get("Positive",0)
        st.metric("Positive", 
                positive_count, 
                f"{positive_count/ len(reviews_df)*100:.2f}%")

    with col2:
        # Show the number of positives rvviews and the percentage
        negative_count = sentiment_counts.get("Negative",0)
        st.metric("Negative", 
                negative_count, 
                f"{negative_count/ len(reviews_df)*100:.2f}%")
        
    with col3:
        # Show the number of positives rvviews and the percentage
        neutral_count = sentiment_counts.get("Neutral",0)
        st.metric("Neutral", 
                neutral_count, 
                f"{neutral_count/ len(reviews_df)*100:.2f}%")
        
    
    # Display pie chart
    import plotly.express as px
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution'
    )
    st.plotly_chart(fig)




# Example usage
# write the ressult to the app with title "Sentiment"
# st.title("Sentiment")
# review_sentiment = classify_sentiment_ollama(user_input)
# if review_sentiment == " Neutral":
#     review_sentiment += "👌"
# elif review_sentiment == " Negative":
#     review_sentiment += "😒"
# else:
#     review_sentiment += "😂"
# st.write(review_sentiment)
