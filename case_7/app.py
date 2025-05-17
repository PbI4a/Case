import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

st.title("Простой просмотр отзывов Amazon")

@st.cache_data
def load_data():
    token = st.secrets["HF_TOKEN"]
    path = hf_hub_download(
        repo_id="PbI4a/Case_7",
        filename="clean_reviews.csv",
        repo_type="dataset",
        use_auth_token=token
    )
    df = pd.read_csv(path)
    return df

df = load_data()

rating_filter = st.multiselect(
    "Фильтр по рейтингу (звёзды)",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5]
)

df_filtered = df[df['star_rating'].isin(rating_filter)]

st.write(f"Показано отзывов: {len(df_filtered)}")

st.dataframe(df_filtered.head(10))
