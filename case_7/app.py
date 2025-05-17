import streamlit as st
import pandas as pd
import requests
from io import BytesIO

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

@st.cache_data(show_spinner=True)
def load_data_from_gdrive(file_id):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    file_bytes = BytesIO()
    for chunk in response.iter_content(32768):
        file_bytes.write(chunk)
    file_bytes.seek(0)

    df = pd.read_csv(file_bytes)
    return df

file_id = "1RN4mmaROL1PDP-9_2rCvROnsVFksTMXc"

try:
    df = load_data_from_gdrive(file_id)
    st.write("Колонки в датасете:", df.columns.tolist())
    st.write(df.head())
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()
