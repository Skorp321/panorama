import base64
import contextlib
import io
import os
import sqlite3
import tempfile
import numpy as np

import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detectionApp import detect
import ffmpegcv
import multiprocessing
import streamlit.components.v1 as components
import base64


# Функция для подключения к базе данных и извлечения данных
def get_data_from_db():
    conn = sqlite3.connect("/container_dir/data/soccer_analitics.db")
    query = "SELECT * FROM analytics"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main():

    st.set_page_config(
        page_title="AI Football",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="/container_dir/data/Logo2.png"
    )
    # Добавление пользовательских стилей
    st.markdown(
        """
    <style>
    body {
        background-color: #333;
        color: #ddd;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title(
        "Детекция футбольных игроков с разбиением на команды и проекцией на тактическую карту."
    )
    st.subheader(":red[Хорошо работает только с панорамным видео!]")

    url = 'https://tl-razrabotka.ru/'
    # Путь к картинке
    image_path = "/container_dir/data/Logo2.png"

    # HTML код для отображения картинки с ссылкой
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # HTML код для отображения картинки с ссылкой
    html_code = f'''
    <a href="{url}" target="_blank">
        <img src="data:image/png;base64,{encoded_string}" alt="My Image" style="width:90%;height:auto;">
    </a>
    '''

    # Встраивание HTML кода в Streamlit приложение
    st.sidebar.markdown(html_code, unsafe_allow_html=True)       
    #st.sidebar.image("/container_dir/data/Logo2.svg", width=250)
    
    st.sidebar.title("Основные настройки")
    
    demo_selected = st.sidebar.radio(
        label="Выберите демо видео:", options=["Пример 1", "Пример 2"], horizontal=True
    )

    ## Sidebar Setup
    st.sidebar.markdown("---")
    st.sidebar.subheader("Загрузка видео")
    input_vide_file = st.sidebar.file_uploader(
        "Загрузите видео файл", type=["mp4", "mov", "avi", "m4v", "asf"]
    )

    team1_name = 'first'
    team2_name = 'second'

    demo_vid_paths = {
        "Пример 1": "/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4",
        "Пример 2": "/container_dir/data/Swiss_vs_slovakia-Panorama.mp4",
    }
    demo_vid_path = demo_vid_paths[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Предпросмотр видео:')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)

    st.sidebar.markdown("---")

    ## Page Setup
    tab1, tab3 = st.tabs(["Как запустить?", "Настройки детекции"])
    with tab1:
        #st.header(":blue[Welcome!]")
        st.subheader("Основные возможности приложения:", divider="blue")
        st.markdown(
            """
                    1. Детекция игроков и судей.
                    2. Разбиение игроков на команды.
                    3. Отображение положения игроков на тактической карте.
                    """
        )
        st.subheader("Как запустить?", divider="blue")
        st.markdown(
            """
                    **Есть два демонстрационных видеоролика, которые автоматически загружаются при запуске приложения, а также рекомендуемые настройки и гиперпараметры**
                    1. Загрузите видео для анализа, воспользовавшись кнопкой "Обзор файлов" в боковом меню.                    
                    2. Перейдите на вкладку "Настройки детекции", выберите параметры аннотации.
                    3. Запустите детекцию!
                    4. По завершению анализа или при нажатии кнопки "Остановить анализ" появится кнопка для скачивания результатов анализа.
                    """
        )
        #st.write("Version 0.0.1")

    with tab3:
        t2col1, t2col, t2col2 = st.columns([1, 0.7, 0.7])
        with t2col1:
            st.subheader("Данные детекции и трекинга:")
            df_field = st.empty()

        with t2col:
            st.subheader("Аннотация таблицы:")
            st.write("frame - номер кадра.")
            st.write("x_anchor, y_anchor - координаты x и y игрока на поле.")
            st.write("team - принадлежность игрока к команде 1 или 2.")
            st.write("id - номер трека игрока.")
            st.write("cls - класс детекции(1 - игрок, 2 - судья)")
            st.write("conf - уверенность модели в детекции(max = 1)")
            
        with t2col2:
            st.write("Опции визуализации:")
            #bcol21t, bcol22t = st.columns([1, 1])
            #with bcol21t:
            show_k = st.toggle(label="Отобразить ключевые точки", value=False)
            show_p = st.toggle(label="Отобразить детекции игроков", value=True)
            show_pal = st.toggle(label="Отобразить трэки", value=True)
            #with bcol22t:
                
            show_b = False # st.toggle(label="Отобразить детекцию мяча", value=False)
            plot_hyperparser = {0: show_k, 1: show_pal, 2: show_b, 3: show_p}

            st.markdown("---")
            save_output =  False #st.checkbox(label="Путь сохранения", value=False)
            output_file_name = None

        bcol1, bcol2 = st.columns([1, 0.7])
        with bcol1:
            st.write("")
        with bcol2:
            # Инициализация сессионного состояния для кнопок
            if "start_pressed" not in st.session_state:
                st.session_state.start_pressed = False

            if "stop_pressed" not in st.session_state:
                st.session_state.stop_pressed = False

            if 'download_ready' not in st.session_state:
                st.session_state.download_ready = False

            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 1, 1, 1])

            with bcol22:
                ready = team1_name == "" or team2_name == ""
                start_detection = st.button(label="Начать анализ", disabled=ready)
                if start_detection:
                    st.session_state.start_pressed = True
            with bcol23:
                stop_btn_state = not st.session_state.start_pressed
                stop_detection = st.button(
                    label="Остановить анализ", disabled=stop_btn_state
                )
                if stop_detection:
                    st.session_state.stop_pressed = True
            with bcol21:
                flag = not (
                    st.session_state.start_pressed and st.session_state.stop_pressed
                )

                if not flag:
                    data = get_data_from_db()

                    # Конвертация в csv
                    csv_buffer = io.StringIO()
                    data.to_csv(csv_buffer, index=False)

                    csv_bytes = csv_buffer.getvalue().encode("utf-8")

                    st.download_button(label="Скачать файл", data=csv_bytes, file_name="data.csv", mime="text/csv")


            with bcol24:
                st.write("")
        st.markdown("---")
    stframe = st.empty()
    cap = ffmpegcv.VideoCapture(tempf.name)
    status = False

    if start_detection and not stop_detection:
        st.toast("Detection Started!")
        parent_dir = "/container_dir/Deep-EIoU/Deep-EIoU"

        # Изменение текущего рабочего каталога
        os.chdir(parent_dir)
        status = detect(
            cap, stframe, output_file_name, save_output, plot_hyperparser, df_field
        )

    else:
        with contextlib.suppress(Exception):
            # Release the video capture object and close the display window
            cap.release()
    if status:
        st.toast("Detection Completed!")
        cap.release()

        data = get_data_from_db()

        # Конвертация в csv
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)

        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(label="Скачать файл", data=csv_bytes, file_name="data.csv", mime="text/csv")


import contextlib

if __name__ == "__main__":
    with contextlib.suppress(SystemExit):
        main()
