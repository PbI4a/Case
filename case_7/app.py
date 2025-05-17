import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO

@st.cache_data
def load_data_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content))
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.dropna(subset=['review_date'])
    return df

file_id = "1RN4mmaROL1PDP-9_2rCvROnsVFksTMXc"

try:
    df = load_data_from_gdrive(file_id)
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

st.title("Анализ настроений в отзывах о продуктах")

st.sidebar.header("Фильтры")

try:
    cats = df['product_category'].dropna().unique().tolist()
except Exception as e:
    cats = []
    st.sidebar.error(f"Ошибка при получении категорий: {e}")

category = st.sidebar.selectbox("Категория товара", ["Все"] + cats)

rates = {
    "Все": None,
    "Положительные (4-5*)": "Положительный отзыв (4* и 5*)",
    "Нейтральные (3*)": "Нейтральный отзыв (3*)",
    "Негативные (1-2*)": "Негативный отзыв (1* и 2*)"
}
rating_choice = st.sidebar.selectbox("Категория рейтинга", list(rates.keys()))

try:
    start_date, end_date = st.sidebar.date_input(
        "Диапазон дат", [df['review_date'].min().date(), df['review_date'].max().date()]
    )
except Exception as e:
    st.sidebar.error(f"Ошибка выбора дат: {e}")
    start_date = df['review_date'].min().date()
    end_date = df['review_date'].max().date()

dff = df.copy()
try:
    if category != "Все":
        dff = dff[dff['product_category'] == category]
    if rates[rating_choice]:
        dff = dff[dff['star_rating_category'] == rates[rating_choice]]
    dff = dff[(dff['review_date'] >= pd.to_datetime(start_date)) & (dff['review_date'] <= pd.to_datetime(end_date))]
except Exception as e:
    st.error(f"Ошибка фильтрации данных: {e}")
    dff = df.copy()

if dff.empty:
    st.warning("Нет данных по выбранным фильтрам.")
else:
    st.subheader("Топ-10 категорий товаров")
    top_cats = dff['product_category'].value_counts().nlargest(10).reset_index()
    top_cats.columns = ['Категория', 'Количество']
    st.plotly_chart(px.bar(top_cats, x='Категория', y='Количество'), use_container_width=True)

    st.subheader("Распределение отзывов по настроению")
    cnt = dff['star_rating_category'].value_counts().reset_index()
    cnt.columns = ['Настроение', 'Количество']
    st.plotly_chart(px.pie(cnt, names='Настроение', values='Количество'), use_container_width=True)

    st.subheader("Динамика среднего рейтинга во времени")
    trend = dff.groupby('review_date')['star_rating'].mean().reset_index()
    st.plotly_chart(px.line(trend, x='review_date', y='star_rating',
                            labels={'review_date': 'Дата', 'star_rating': 'Средний рейтинг'}),
                    use_container_width=True)

    st.subheader("Распределение длины отзывов")
    st.plotly_chart(px.histogram(dff, x='word_count', nbins=50), use_container_width=True)

    if rates[rating_choice]:
        st.subheader(f"Топ-25 слов для: {rating_choice}")
        words = dff['review_body'].str.split().explode()
        topw = words.value_counts().nlargest(25).reset_index()
        topw.columns = ['Слово', 'Частота']
        st.plotly_chart(px.bar(topw, x='Частота', y='Слово', orientation='h'), use_container_width=True)
