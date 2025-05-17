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

# 0. Настройка страницы
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Amazon US Product Reviews Dashboard")
st.markdown("Загрузка и анализ отзывов из датасета на Hugging Face (3 ГБ) через hf_hub_download().")

# 1. Загрузка и предобработка данных
@st.cache_data(show_spinner=True)
def load_data():
    token = st.secrets["HF_TOKEN"]
    file_path = hf_hub_download(
        repo_id="PbI4a/Case_7",
        filename="clean_reviews.csv",
        repo_type="dataset",
        use_auth_token=token
    )
    df = pd.read_csv(file_path)
    # Преобработка
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['year_month'] = df['review_date'].dt.to_period('M').astype(str)
    df['sentiment_simple'] = df['star_rating'].apply(
        lambda r: 'negative' if r <= 2 else 'neutral' if r == 3 else 'positive'
    )
    df['polarity'] = df['review_body'].fillna("").map(
        lambda t: TextBlob(t).sentiment.polarity
    )
    return df

with st.spinner("Загружаем и подготавливаем данные…"):
    df = load_data()

# 2. Глобальные фильтры
st.sidebar.header("🔍 Фильтры")
years = sorted(df['review_date'].dt.year.dropna().unique().astype(int))
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
    df['product_title'].dropna().unique(),
    max_selections=5,
    default=df['product_title'].dropna().unique()[:3]
)
verified = st.sidebar.checkbox("Только подтверждённые покупки", value=False)

mask = (
    (df['review_date'].dt.year >= yr_min) &
    (df['review_date'].dt.year <= yr_max) &
    (df['star_rating'].isin(ratings))
)
if products:
    mask &= df['product_title'].isin(products)
if verified:
    mask &= df['verified_purchase'] == 'Y'
df_filtered = df[mask]

# 3. Навигационные вкладки
tabs = st.tabs([
    "Overview", "Sentiment", "Time Series", "Comparison",
    "Text Analysis", "Clustering", "Topic Modeling", "Reviews", "Export"
])

# --- Overview
with tabs[0]:
    st.header("📊 Обзор")
    c1, c2, c3 = st.columns(3)
    c1.metric("Отзывы всего", f"{len(df_filtered):,}")
    c2.metric("Средний рейтинг", round(df_filtered['star_rating'].mean(), 2))
    c3.metric("Средняя полярность", round(df_filtered['polarity'].mean(), 2))
    fig = px.histogram(
        df_filtered,
        x='star_rating',
        nbins=5,
        title="Распределение рейтингов"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sentiment
with tabs[1]:
    st.header("💬 Тональность")
    simple = df_filtered['sentiment_simple'].value_counts().reset_index()
    simple.columns = ['sentiment', 'count']
    fig1 = px.pie(
        simple,
        names='sentiment',
        values='count',
        title="По рейтингу",
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    )
    fig2 = px.histogram(
        df_filtered,
        x='polarity',
        nbins=30,
        title="TextBlob полярность"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --- Time Series
with tabs[2]:
    st.header("⏳ Динамика во времени")
    grp = df_filtered.groupby('year_month').agg(
        reviews=('star_rating', 'count'),
        avg_rating=('star_rating', 'mean')
    ).reset_index()
    fig1 = px.bar(grp, x='year_month', y='reviews', title="Отзывы по месяцам")
    fig2 = px.line(grp, x='year_month', y='avg_rating', title="Средний рейтинг")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --- Comparison
with tabs[3]:
    st.header("🔀 Сравнение продуктов")
    sel = st.multiselect(
        "Продукты для сравнения",
        products,
        default=products[:3],
        max_selections=5
    )
    comp = df_filtered[df_filtered['product_title'].isin(sel)].groupby(
        ['product_title', 'year_month']
    )['star_rating'].mean().reset_index()
    fig = px.line(
        comp,
        x='year_month',
        y='star_rating',
        color='product_title',
        title="Сравнение средней оценки"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Text Analysis
with tabs[4]:
    st.header("📝 Текстовый анализ")
    sel_s = st.selectbox("Тональность для облака", ['positive', 'neutral', 'negative'])
    txt = " ".join(df_filtered[df_filtered['sentiment_simple'] == sel_s]['review_body'].dropna())
    wc = WordCloud(width=800, height=300, background_color="white").generate(txt or " ")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Топ‑20 слов и биграм")
    tokens = pd.Series(" ".join(df_filtered['review_body'].dropna()).lower().split())
    topw = tokens.value_counts().head(20).reset_index()
    topb = pd.Series(zip(tokens, tokens.shift(-1))).value_counts().head(20).reset_index()
    topb.columns = ['bigram', 'count']
    fig1 = px.bar(topw, x='count', y='index', orientation='h', title="Топ-20 слов")
    fig2 = px.bar(topb, x='count', y='bigram', orientation='h', title="Топ-20 биграм")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --- Clustering
with tabs[5]:
    st.header("🔍 Кластеризация отзывов")
    n_clusters = st.slider("Число кластеров", 2, 10, 4)
    sample_size = min(len(df_filtered), 50_000)
    df_sample = df_filtered.sample(sample_size, random_state=42)
    vect = TfidfVectorizer(max_features=5_000, stop_words='english')
    X = vect.fit_transform(df_sample['review_body'].fillna(""))
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    df_sample['cluster'] = km.labels_
    cnt = df_sample['cluster'].value_counts().sort_index().reset_index()
    cnt.columns = ['cluster', 'count']
    fig = px.bar(cnt, x='cluster', y='count', title="Кластеры по размеру")
    st.plotly_chart(fig, use_container_width=True)

    terms = vect.get_feature_names_out()
    order = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(n_clusters):
        st.markdown(f"**Кластер {i}:** " + ", ".join(terms[idx] for idx in order[i, :10]))

# --- Topic Modeling
with tabs[6]:
    st.header("📂 Тематическое моделирование (LDA)")
    n_topics = st.slider("Число тем", 2, 10, 4)
    sample_size_lda = min(len(df_filtered), 20_000)
    df_lda = df_filtered.sample(sample_size_lda, random_state=42)
    vect2 = TfidfVectorizer(max_features=2_000, stop_words='english')
    X2 = vect2.fit_transform(df_lda['review_body'].fillna(""))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X2)
    terms2 = vect2.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_terms = [terms2[i] for i in topic.argsort()[-10:][::-1]]
        st.markdown(f"**Тема {idx}:** " + ", ".join(top_terms))

# --- Reviews (пагинация)
with tabs[7]:
    st.header("📄 Просмотр отзывов")
    page_size = st.number_input("Отзывы на страницу", min_value=5, max_value=50, value=10, step=5)
    total_pages = (len(df_filtered) - 1) // page_size + 1
    page = st.number_input("Страница", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    subset = df_filtered.iloc[start:start + page_size][['review_date', 'product_title', 'star_rating', 'review_body']]
    for _, row in subset.iterrows():
        with st.expander(f"{row['review_date'].date()} | ⭐{row['star_rating']} | {row['product_title'][:30]}"):
            st.write(row['review_body'])

# --- Export
with tabs[8]:
    st.header("💾 Экспорт данных")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать CSV", data=csv, file_name="filtered_reviews.csv", mime="text/csv")
    html = df_filtered.to_html()
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;charset=utf-8;base64,{b64}" download="reviews.html">Скачать HTML-отчёт</a>'
    st.markdown(href, unsafe_allow_html=True)
