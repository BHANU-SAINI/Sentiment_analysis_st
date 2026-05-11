import streamlit as st
import pickle
import re
import requests
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import plotly.express as px

from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

# -----------------------------
# Cache resources
# -----------------------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    nltk.download('punkt')
    return stopwords.words('english')


@st.cache_resource
def load_model_and_vectorizer():

    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


@st.cache_resource
def initialize_youtube():

    from googleapiclient.discovery import build

    api_key = st.secrets["YOUTUBE_API_KEY"]

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    return youtube


# -----------------------------
# Preprocess and predict
# -----------------------------
def predict_sentiment_with_score(text, model, vectorizer, stop_words):

    text_proc = re.sub('[^a-zA-Z]', ' ', text).lower().split()

    text_proc = [
        word for word in text_proc
        if word not in stop_words
    ]

    text_proc = ' '.join(text_proc)

    vect_text = vectorizer.transform([text_proc])

    pred = model.predict(vect_text)[0]

    sentiment_label = "Positive" if pred == 1 else "Negative"

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.5:
        sentiment_label = "Positive"

    elif polarity < -0.5:
        sentiment_label = "Negative"

    return sentiment_label, polarity


# -----------------------------
# Main App
# -----------------------------
def main():

    st.title("Sentiment Analysis App")

    stop_words = load_stopwords()

    model, vectorizer = load_model_and_vectorizer()

    youtube = initialize_youtube()

    option = st.selectbox(
        "Choose an option",
        [
            "Input text",
            "Get posts from subreddit",
            "Get YouTube video comments"
        ]
    )

    results = []

    # -----------------------------
    # 1️⃣ Manual Text Input
    # -----------------------------
    if option == "Input text":

        text_input = st.text_area(
            "Enter text to analyze sentiment"
        )

        if st.button("Analyze"):

            sentiment, polarity = predict_sentiment_with_score(
                text_input,
                model,
                vectorizer,
                stop_words
            )

            results.append({
                "Text": text_input,
                "Sentiment": sentiment,
                "Polarity": polarity
            })

            st.write(f"### Sentiment: {sentiment}")

            st.write(f"### Polarity: {polarity:.2f}")

    # -----------------------------
    # 2️⃣ Reddit Posts Fetch
    # -----------------------------
    elif option == "Get posts from subreddit":

        subreddit_name = st.text_input(
            "Enter subreddit name"
        )

        if st.button("Fetch Posts"):

            try:

                headers = {
                    "User-Agent":
                    "sentiment-analysis-app by u/YOUR_REDDIT_USERNAME"
                }

                url = (
                    f"https://www.reddit.com/r/"
                    f"{subreddit_name}/hot.json?limit=50"
                )

                response = requests.get(
                    url,
                    headers=headers
                )

                if response.status_code != 200:

                    st.error(
                        f"Error: Received status code "
                        f"{response.status_code}"
                    )

                else:

                    data = response.json()

                    posts = []

                    for post in data["data"]["children"]:

                        title = post["data"].get(
                            "title",
                            ""
                        )

                        selftext = post["data"].get(
                            "selftext",
                            ""
                        )

                        text_combined = (
                            f"{title} {selftext}"
                        ).strip()

                        posts.append(text_combined)

                    if not posts:

                        st.warning("No posts found.")

                    else:

                        for post_text in posts:

                            sentiment, polarity = (
                                predict_sentiment_with_score(
                                    post_text,
                                    model,
                                    vectorizer,
                                    stop_words
                                )
                            )

                            results.append({
                                "Text": post_text,
                                "Sentiment": sentiment,
                                "Polarity": polarity
                            })

                            st.write(
                                f"### Sentiment: {sentiment}"
                            )

                            st.write(
                                f"### Polarity: "
                                f"{polarity:.2f}"
                            )

                            st.write(post_text)

                            st.write("---")

                        st.session_state["results"] = results

            except Exception as e:

                st.error(
                    f"Error fetching posts: {e}"
                )

    # -----------------------------
    # 3️⃣ YouTube Comments Fetch
    # -----------------------------
    elif option == "Get YouTube video comments":

        video_id = st.text_input(
            "Enter YouTube Video ID"
        )

        if st.button("Fetch Comments"):

            try:

                comments = []

                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText"
                )

                while request:

                    response = request.execute()

                    for item in response['items']:

                        comment = item[
                            'snippet'
                        ][
                            'topLevelComment'
                        ][
                            'snippet'
                        ][
                            'textDisplay'
                        ]

                        comments.append(comment)

                    request = (
                        youtube.commentThreads().list_next(
                            request,
                            response
                        )
                    )

                if not comments:

                    st.warning("No comments found.")

                else:

                    for comment_text in comments:

                        sentiment, polarity = (
                            predict_sentiment_with_score(
                                comment_text,
                                model,
                                vectorizer,
                                stop_words
                            )
                        )

                        results.append({
                            "Text": comment_text,
                            "Sentiment": sentiment,
                            "Polarity": polarity
                        })

                        st.write(
                            f"### Sentiment: {sentiment}"
                        )

                        st.write(
                            f"### Polarity: "
                            f"{polarity:.2f}"
                        )

                        st.write(comment_text)

                        st.write("---")

                    st.session_state["results"] = results

            except Exception as e:

                st.error(
                    f"Error fetching comments: {e}"
                )

    # -----------------------------
    # 📊 Insights Section
    # -----------------------------
    st.markdown("---")

    if st.button("📊 Show Overall Insights"):

        if (
            "results" in st.session_state
            and st.session_state["results"]
        ):

            df = pd.DataFrame(
                st.session_state["results"]
            )

            sentiment_counts = (
                df["Sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_counts.columns = [
                "Sentiment",
                "Count"
            ]

            # -----------------------------
            # Bar Chart
            # -----------------------------
            st.subheader(
                "Sentiment Distribution"
            )

            fig_bar = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                text="Count",
                title="Overall Sentiment Distribution"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

            # -----------------------------
            # Pie Chart
            # -----------------------------
            st.subheader(
                "Sentiment Breakdown"
            )

            fig_pie = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Sentiment Percentage Split"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

            # -----------------------------
            # WordCloud
            # -----------------------------
            st.subheader(
                "Most Frequent Words"
            )

            all_text = " ".join(
                df["Text"].astype(str)
            )

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(all_text)

            fig_wc, ax = plt.subplots()

            ax.imshow(
                wordcloud,
                interpolation="bilinear"
            )

            ax.axis("off")

            st.pyplot(fig_wc)

        else:

            st.warning(
                "Please fetch comments or posts first."
            )


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
