import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# 🔗 Ваша прямая ссылка
CSV_URL = "https://huggingface.co/datasets/PbI4a/Case_7/resolve/main/case_7/clean_reviews.csv"

# 👇 Укажите нужные колонки, чтобы сэкономить память
USECOLS = ['product_title', 'star_rating', 'review_body', 'review_date', 'verified_purchase']

# ⚙️ Кэш-функция для чтения с батчами
@st.cache_data(show_spinner=True)
def load_data(url, usecols, chunksize=500_000):
    chunks = []
    for chunk in pd.read_csv(url, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

st.title("📊 Анализ отзывов Amazon US")

with st.spinner("Загружаем и обрабатываем отзывы..."):
    df = load_data(CSV_URL, usecols=USECOLS)

# 🎯 Интерфейс фильтрации
with st.sidebar:
    st.header("Фильтры")
    min_year, max_year = df['review_date'].str[:4].dropna().astype(int).agg(['min', 'max'])
    selected_year = st.slider("Год отзыва", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    verified_only = st.checkbox("Только подтвержденные покупки", value=False)

# ⛏️ Применение фильтров
df['year'] = df['review_date'].str[:4].astype('Int64')
filtered_df = df[
    (df['year'] >= selected_year[0]) &
    (df['year'] <= selected_year[1])
]

if verified_only:
    filtered_df = filtered_df[filtered_df['verified_purchase'] == 'Y']

# 📈 Пример анализа: распределение оценок
st.subheader("Распределение звёздных оценок")
rating_counts = filtered_df['star_rating'].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax, palette="viridis")
ax.set_xlabel("Оценка")
ax.set_ylabel("Количество отзывов")
st.pyplot(fig)

# 📝 Пример сторителлинга: вывод 5 примеров отзывов
st.subheader("Примеры отзывов")
for i, row in filtered_df.sample(5, random_state=42).iterrows():
    st.markdown(f"**⭐️ {row['star_rating']} | `{row['product_title']}`**")
    st.write(row['review_body'])
    st.markdown("---")
