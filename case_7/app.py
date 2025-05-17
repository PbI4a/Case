import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import hf_hub_download
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import base64

# Настройка страницы
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Amazon US Product Reviews Dashboard")
st.markdown("Загрузка и анализ отзывов из датасета на Hugging Face (3 ГБ) с ленивой загрузкой и чтением батчами.")

# 1. Получение файла с Hugging Face с кэшированием
@st.cache_data(show_spinner=True)
def get_csv_file():
    token = st.secrets["HF_TOKEN"]
    path = hf_hub_download(
        repo_id="PbI4a/Case_7",
        filename="clean_reviews.csv",
        repo_type="dataset",
        use_auth_token=token
    )
    return path

file_path = get_csv_file()

# 2. Получение списка годов и продуктов для фильтров (первый сэмпл)
@st.cache_data(show_spinner=True)
def sample_for_filters(file_path):
    sample = pd.read_csv(file_path, nrows=200_000, parse_dates=['review_date'])
    years = sorted(sample['review_date'].dt.year.dropna().unique())
    products = sample['product_title'].dropna().unique()
    return years, products

years, all_products = sample_for_filters(file_path)

# 3. Интерфейс фильтров
st.sidebar.header("🔍 Фильтры")

yr_min, yr_max = st.sidebar.select_slider(
    "Год отзыва",
    options=years,
    value=(years[0], years[-1])
)

ratings = st.sidebar.multiselect(
    "Рейтинг (звёзды)",
    [1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5]
)

products = st.sidebar.multiselect(
    "Продукты (до 5)",
    all_products,
    max_selections=5,
    default=list(all_products[:3])
)

verified = st.sidebar.checkbox("Только подтверждённые покупки", value=False)

filters = {
    'year_min': yr_min,
    'year_max': yr_max,
    'ratings': ratings,
    'products': products if products else None,
    'verified': verified
}

# 4. Функция чтения CSV батчами с фильтрами
def read_filtered_chunks(file_path, filters, chunksize=100_000):
    for chunk in pd.read_csv(file_path, chunksize=chunksize, parse_dates=['review_date']):
        chunk['year'] = chunk['review_date'].dt.year
        chunk['sentiment_simple'] = chunk['star_rating'].apply(
            lambda r: 'negative' if r <= 2 else 'neutral' if r == 3 else 'positive'
        )
        chunk['polarity'] = chunk['review_body'].fillna("").map(lambda t: TextBlob(t).sentiment.polarity)

        mask = (
            (chunk['year'] >= filters['year_min']) &
            (chunk['year'] <= filters['year_max']) &
            (chunk['star_rating'].isin(filters['ratings']))
        )
        if filters.get('products'):
            mask &= chunk['product_title'].isin(filters['products'])
        if filters.get('verified'):
            mask &= chunk['verified_purchase'] == 'Y'

        filtered = chunk.loc[mask]
        if not filtered.empty:
            yield filtered

# 5. Агрегируем отфильтрованные чанки
@st.cache_data(show_spinner=True)
def aggregate_filtered(file_path, filters):
    df_list = []
    for filtered_chunk in read_filtered_chunks(file_path, filters):
        df_list.append(filtered_chunk)
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame(columns=[
            'review_date', 'star_rating', 'review_body', 'product_title',
            'verified_purchase', 'sentiment_simple', 'polarity', 'year'
        ])
    return df

df_filtered = aggregate_filtered(file_path, filters)

# 6. Обзор (Overview)
st.header("📊 Обзор")

c1, c2, c3 = st.columns(3)
c1.metric("Отзывы всего", f"{len(df_filtered):,}")
c2.metric("Средний рейтинг", round(df_filtered['star_rating'].mean(), 2) if not df_filtered.empty else "N/A")
c3.metric("Средняя полярность", round(df_filtered['polarity'].mean(), 2) if not df_filtered.empty else "N/A")

if not df_filtered.empty:
    fig = px.histogram(
        df_filtered,
        x='star_rating',
        nbins=5,
        title="Распределение рейтингов"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Нет данных для отображения по выбранным фильтрам.")

# 7. Временной ряд отзывов по месяцам
st.header("📈 Динамика отзывов по месяцам")
if not df_filtered.empty:
    df_filtered['month'] = df_filtered['review_date'].dt.to_period('M').dt.to_timestamp()
    ts = df_filtered.groupby('month').size().reset_index(name='count')
    fig_ts = px.line(ts, x='month', y='count', title="Количество отзывов по месяцам")
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("Нет данных для построения временного ряда.")

# 8. Облако слов (WordCloud)
st.header("☁️ Облако слов из отзывов")

if not df_filtered.empty:
    text = " ".join(df_filtered['review_body'].dropna().astype(str).values)
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Недостаточно текста для облака слов.")
else:
    st.info("Нет данных для облака слов.")

# 9. Кластеризация отзывов по TF-IDF и KMeans (на сэмпле)
st.header("🧩 Кластеризация отзывов (на сэмпле)")

if len(df_filtered) > 1000:
    sample_size = 2000
else:
    sample_size = len(df_filtered)

if sample_size > 0:
    sample_reviews = df_filtered['review_body'].dropna().sample(sample_size, random_state=42).astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(sample_reviews)

    n_clusters = 5
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(X)

    df_clusters = pd.DataFrame({'review': sample_reviews, 'cluster': clusters})

    # Выводим количество отзывов по кластерам
    cluster_counts = df_clusters['cluster'].value_counts().sort_index()
    fig_cluster = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Кластер', 'y': 'Количество отзывов'},
        title="Количество отзывов по кластерам"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Показываем первые 3 отзыва из каждого кластера
    for i in range(n_clusters):
        st.subheader(f"Кластер {i}")
        for review_text in df_clusters[df_clusters['cluster'] == i]['review'].head(3):
            st.write(f"- {review_text[:300]}{'...' if len(review_text) > 300 else ''}")
else:
    st.info("Недостаточно данных для кластеризации.")

# 10. Тематическое моделирование LDA (на сэмпле)
st.header("📚 Тематическое моделирование (LDA)")

if sample_size > 0:
    vectorizer_lda = TfidfVectorizer(stop_words='english', max_features=1000)
    X_lda = vectorizer_lda.fit_transform(sample_reviews)

    n_topics = 5
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_lda)

    feature_names = vectorizer_lda.get_feature_names_out()

    for topic_idx, topic in enumerate(lda.components_):
        st.subheader(f"Тема {topic_idx + 1}")
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        st.write(", ".join(top_words))
else:
    st.info("Недостаточно данных для тематического моделирования.")

# 11. Просмотр отзывов с пагинацией
st.header("📄 Просмотр отзывов")

page_size = st.number_input("Отзывы на страницу", min_value=5, max_value=50, value=10, step=5)
total_pages = (len(df_filtered) - 1) // page_size + 1 if len(df_filtered) > 0 else 1
page = st.number_input("Страница", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * page_size
subset = df_filtered.iloc[start:start + page_size][['review_date', 'product_title', 'star_rating', 'review_body']]

for _, row in subset.iterrows():
    with st.expander(f"{row['review_date'].date()} | ⭐{row['star_rating']} | {row['product_title'][:30]}"):
        st.write(row['review_body'])


# 12. Опционально — функция для экспорта отфильтрованных данных (необязательно)
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(df_filtered)

st.download_button(
    label="⬇️ Скачать отфильтрованные отзывы CSV",
    data=csv_data,
    file_name='filtered_reviews.csv',
    mime='text/csv'
)
