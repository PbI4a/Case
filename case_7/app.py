import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Проверка загрузки данных", layout="centered")

st.title("🔍 Проверка загрузки Amazon Reviews")

# Отладочные сообщения
st.text("🚀 Начинаем загрузку...")

@st.cache_data(show_spinner=True)
def load_sample():
    st.text("📥 Получаем файл с Hugging Face...")
    token = st.secrets["HF_TOKEN"]
    file_path = hf_hub_download(
        repo_id="PbI4a/Case_7",
        filename="clean_reviews.csv",
        repo_type="dataset",
        use_auth_token=token
    )
    st.text("📖 Загружаем первые 1000 строк...")
    df = pd.read_csv(file_path, nrows=1000)
    return df

try:
    df = load_sample()
    st.success("✅ Данные успешно загружены!")

    st.write("**Размер датафрейма:**", df.shape)
    st.write("**Столбцы:**", df.columns.tolist())
    st.write("**Первые строки:**")
    st.dataframe(df.head())

except Exception as e:
    st.error("❌ Ошибка при загрузке данных:")
    st.exception(e)
