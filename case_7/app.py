import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from io import BytesIO
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

# 1. Загрузка данных батчами с кэшем
@st.cache_data(show_spinner=True)
def load_data(url, chunksize=200_000):
    chunks = []
    for chunk in pd.read_csv(url, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    df_ = pd.concat(chunks, ignore_index=True)
    df_['review_date'] = pd.to_datetime(df_['review_date'], errors='coerce')
    df_['year_month'] = df_['review_date'].dt.to_period('M').astype(str)
    # простая тональность
    df_['sentiment_simple'] = df_['star_rating'].apply(
        lambda r: 'negative' if r<=2 else 'neutral' if r==3 else 'positive')
    # TextBlob sentiment polarity
    df_['polarity'] = df_['review_body'].dropna().apply(lambda t: TextBlob(t).sentiment.polarity)
    return df_

CSV_URL = "https://huggingface.co/datasets/PbI4a/Case_7/resolve/main/case_7/clean_reviews.csv"
with st.spinner("Загружаем и подготавливаем данные…"):
    df = load_data(CSV_URL)

# 2. Глобальные фильтры
st.sidebar.header("🔍 Глобальные фильтры")
years = sorted(df['review_date'].dt.year.dropna().astype(int).unique())
yr_min, yr_max = st.sidebar.select_slider("Год", options=years, value=(years[0], years[-1]))
ratings = st.sidebar.multiselect("Рейтинг", [1,2,3,4,5], default=[1,2,3,4,5])
products = st.sidebar.multiselect("Продукты", df['product_title'].dropna().unique(), max_selections=5, default=df['product_title'].dropna().unique()[:3])
verified = st.sidebar.checkbox("Подтверждённые покупки только", value=False)

mask = (
    (df['review_date'].dt.year >= yr_min) &
    (df['review_date'].dt.year <= yr_max) &
    (df['star_rating'].isin(ratings))
)
if products:
    mask &= df['product_title'].isin(products)
if verified:
    mask &= df['verified_purchase']=='Y'
df = df[mask]

# 3. Табуляция
tabs = st.tabs([
    "Overview","Sentiment","Time Series","Comparison",
    "Text Analysis","Clustering","Topic Modeling","Reviews","Export"
])

# --- Overview
with tabs[0]:
    st.header("📊 Обзор")
    c1, c2, c3 = st.columns(3)
    c1.metric("Всего отзывов", f"{len(df):,}")
    c2.metric("Средний рейтинг", round(df['star_rating'].mean(),2))
    perc = 100*len(df[df['verified_purchase']=='Y'])/len(df) if len(df) else 0
    c3.metric("Подтверждённые покупки", f"{perc:.1f}%")
    fig = px.histogram(df, x='star_rating', nbins=5, title="Распределение рейтингов")
    st.plotly_chart(fig, use_container_width=True)

# --- Sentiment
with tabs[1]:
    st.header("💬 Тональность отзывов")
    # простая
    simple = df['sentiment_simple'].value_counts().reset_index()
    simple.columns = ['sentiment','count']
    fig1 = px.pie(simple, names='sentiment', values='count', title="По рейтингу")
    # TextBlob
    fig2 = px.histogram(df, x='polarity', nbins=30, title="TextBlob polarity")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --- Time Series
with tabs[2]:
    st.header("⏳ Динамика во времени")
    grp = df.groupby('year_month').agg(reviews=('star_rating','count'), avg_rating=('star_rating','mean')).reset_index()
    fig1 = px.bar(grp, x='year_month', y='reviews', title="Количество отзывов по месяцам")
    fig2 = px.line(grp, x='year_month', y='avg_rating', title="Средний рейтинг по месяцам")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --- Comparison
with tabs[3]:
    st.header("🔀 Сравнение продуктов")
    sel = st.multiselect("Продукты для сравнения", products, default=products)
    comp = df[df['product_title'].isin(sel)].groupby(['product_title','year_month']).star_rating.mean().reset_index()
    fig = px.line(comp, x='year_month', y='star_rating', color='product_title',
                  title="Средний рейтинг по продуктам")
    st.plotly_chart(fig, use_container_width=True)

# --- Text Analysis
with tabs[4]:
    st.header("📝 Текстовый анализ")
    # WordCloud
    st.subheader("Облако слов по простой тональности")
    sel_s = st.selectbox("Тональность", ['positive','neutral','negative'])
    txt = " ".join(df[df['sentiment_simple']==sel_s]['review_body'].dropna())
    wc = WordCloud(width=800, height=300, background_color="white").generate(txt or " ")
    fig, ax = plt.subplots(figsize=(10,3)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    st.pyplot(fig)
    # Топ‑20 слов и топ‑20 биграм
    st.subheader("Топ‑20 слов и биграм")
    tokens = pd.Series(" ".join(df['review_body'].dropna()).lower().split())
    topw = tokens.value_counts().head(20).reset_index()
    topb = pd.Series(zip(tokens, tokens.shift(-1))).dropna().value_counts().head(20).reset_index()
    topb.columns=['bigram','count']
    fig1 = px.bar(topw, x='count', y='index', orientation='h', title="Топ-20 слов")
    fig2 = px.bar(topb, x='count', y='bigram', orientation='h', title="Топ-20 биграм")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --- Clustering
with tabs[5]:
    st.header("🔍 Кластеризация отзывов")
    n_clusters = st.slider("Число кластеров", 2, 10, 4)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['review_body'].fillna(""))
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    df['cluster'] = km.labels_
    st.subheader("Количество отзывов в кластерах")
    cnt = df['cluster'].value_counts().sort_index().reset_index()
    cnt.columns=['cluster','count']
    fig = px.bar(cnt, x='cluster', y='count', title="Кластеры по размеру")
    st.plotly_chart(fig, use_container_width=True)
    # Топ‑10 терминов каждого кластера
    terms = vectorizer.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(n_clusters):
        st.markdown(f"**Кластер {i} — топ-10 терминов:** " + ", ".join(terms[ind] for ind in order_centroids[i, :10]))

# --- Topic Modeling
with tabs[6]:
    st.header("📂 Тематическое моделирование (LDA)")
    n_topics = st.slider("Число тем", 2, 10, 4)
    vect2 = TfidfVectorizer(max_features=2000, stop_words='english')
    X2 = vect2.fit_transform(df['review_body'].fillna(""))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X2)
    terms2 = vect2.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        st.markdown(f"**Тема {idx}:** " + ", ".join([terms2[i] for i in topic.argsort()[-10:][::-1]]))

# --- Reviews (пагинация)
with tabs[7]:
    st.header("📄 Просмотр отзывов")
    page_size = st.number_input("Отзывы на страницу", 5, 50, 10)
    total = len(df)
    pages = (total // page_size) + 1
    page = st.number_input("Страница", 1, pages, 1)
    start = (page-1)*page_size
    subset = df.iloc[start:start+page_size][['review_date','product_title','star_rating','review_body']]
    for _, row in subset.iterrows():
        with st.expander(f"{row['review_date'].date()} | ⭐️ {row['star_rating']} | {row['product_title'][:30]}"):
            st.write(row['review_body'])

# --- Export
with tabs[8]:
    st.header("💾 Экспорт")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать CSV", data=csv_bytes, file_name="filtered_reviews.csv", mime="text/csv")
    # PDF (упрощённо)
    html = df.to_html() 
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="reviews.html">Скачать HTML-отчёт</a>'
    st.markdown(href, unsafe_allow_html=True)
