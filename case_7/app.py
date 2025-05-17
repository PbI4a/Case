import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# URL на Hugging Face
CSV_URL = "https://huggingface.co/datasets/PbI4a/Case_7/resolve/main/case_7/clean_reviews.csv"

# Настройка страницы
st.set_page_config(page_title="Amazon Reviews Analyzer", layout="wide")

@st.cache_data(show_spinner=True)
def load_data_in_chunks(url, chunksize=100000):
    chunks = []
    for chunk in pd.read_csv(url, chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

# Загрузка данных
st.title("📊 Amazon Product Reviews Analysis")
st.markdown("Анализ отзывов пользователей с использованием фильтров, метрик и визуализации.")

with st.spinner("Загружаем данные..."):
    df = load_data_in_chunks(CSV_URL)

# --- Проверка колонок
expected_columns = ["product_title", "star_rating", "review_date", "review_body", "verified_purchase"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    st.error(f"Отсутствуют необходимые столбцы: {missing_columns}")
    st.stop()

# --- Фильтры
st.sidebar.header("🔍 Фильтры")

# Уникальные значения
product_options = sorted(df["product_title"].dropna().unique().tolist())
selected_products = st.sidebar.multiselect("Выберите продукт(ы)", product_options, default=product_options[:3])

rating_options = sorted(df["star_rating"].dropna().unique().astype(int).tolist())
selected_ratings = st.sidebar.multiselect("Выберите рейтинг", rating_options, default=rating_options)

verified_only = st.sidebar.checkbox("Только подтверждённые покупки", value=False)

# --- Применение фильтров
filtered_df = df[
    df["product_title"].isin(selected_products) &
    df["star_rating"].isin(selected_ratings)
]

if verified_only:
    filtered_df = filtered_df[filtered_df["verified_purchase"] == "Y"]

st.markdown(f"### 📦 Количество отфильтрованных отзывов: {len(filtered_df):,}")

# --- Метрики
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📈 Средняя оценка", round(filtered_df["star_rating"].mean(), 2))
with col2:
    st.metric("📝 Всего отзывов", f"{len(filtered_df):,}")
with col3:
    percent_verified = 100 * len(filtered_df[filtered_df["verified_purchase"] == "Y"]) / len(filtered_df) if len(filtered_df) > 0 else 0
    st.metric("✅ Подтверждённые покупки", f"{percent_verified:.1f}%")

# --- Визуализации
st.subheader("📊 Распределение оценок")
fig_hist = px.histogram(filtered_df, x="star_rating", nbins=5, title="Распределение звёзд")
st.plotly_chart(fig_hist, use_container_width=True)

# --- Wordcloud
st.subheader("☁️ Облако слов")
text_data = " ".join(filtered_df["review_body"].dropna().astype(str).tolist())

if text_data:
    wordcloud = WordCloud(width=1000, height=400, background_color="white").generate(text_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("Недостаточно данных для генерации облака слов.")

# --- Отображение отзывов
st.subheader("🗂 Примеры отзывов")
sample_size = min(10, len(filtered_df))
if sample_size > 0:
    st.dataframe(filtered_df[["review_date", "product_title", "star_rating", "review_body"]].sample(sample_size))
else:
    st.warning("Нет отзывов, соответствующих выбранным фильтрам.")
