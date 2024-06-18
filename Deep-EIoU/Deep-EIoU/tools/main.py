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


# Функция для подключения к базе данных и извлечения данных
def get_data_from_db():
    conn = sqlite3.connect("/container_dir/data/soccer_analitics.db")
    query = "SELECT * FROM analytics"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main():

    flag_1, flag_2 = False, False

    st.set_page_config(
        page_title="AI Powered Web Application for Football Tactical Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
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

    st.sidebar.title("Основные настройки")
    demo_selected = st.sidebar.radio(
        label="Выбирите демо видео:", options=["Demo 1", "Demo 2"], horizontal=True
    )

    ## Sidebar Setup
    st.sidebar.markdown("---")
    st.sidebar.subheader("Загрузка видео")
    input_vide_file = st.sidebar.file_uploader(
        "Загрузите видео файл", type=["mp4", "mov", "avi", "m4v", "asf"]
    )

    demo_vid_paths = {
        "Demo 1": "/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4",
        "Demo 2": "/container_dir/data/Swiss_vs_slovakia-Panorama.mp4",
    }
    demo_vid_path = demo_vid_paths[demo_selected]
    demo_team_info = {
        "Demo 1": {
            "team1_name": "France",
            "team2_name": "Switzerland",
            "team1_p_color": "#1E2530",
            "team1_gk_color": "#F5FD15",
            "team2_p_color": "#FBFCFA",
            "team2_gk_color": "#B1FCC4",
        },
        "Demo 2": {
            "team1_name": "Chelsea",
            "team2_name": "Manchester City",
            "team1_p_color": "#29478A",
            "team1_gk_color": "#DC6258",
            "team2_p_color": "#90C8FF",
            "team2_gk_color": "#BCC703",
        },
    }
    selected_team_info = demo_team_info[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
    else:
        tempf.write(input_vide_file.read())
    demo_vid = open(tempf.name, "rb")
    demo_bytes = demo_vid.read()

    st.sidebar.text("Демо видео:")
    st.sidebar.video(demo_bytes)
    # Load the YOLOv8 players detection model
    model_players = YOLO("../../models/yolov8m_goalkeeper_1280.pt")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("../../models/Yolo8M Field Keypoints/weights/best.pt")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Названия команд")
    team1_name = st.sidebar.text_input(
        label="Имя первой команды", value=selected_team_info["team1_name"]
    )
    team2_name = st.sidebar.text_input(
        label="Имя второй команды", value=selected_team_info["team2_name"]
    )
    st.sidebar.markdown("---")

    ## Page Setup
    tab1, tab3 = st.tabs(["Как запустить?", "Настройки детекции"])
    with tab1:
        st.header(":blue[Welcome!]")
        st.subheader("Основные возможности приложения:", divider="blue")
        st.markdown(
            """
                    1. Детекция игроков, судей и мяча.
                    2. Разбиение игроков на команды.
                    3. Отображение положения игроков и мяча на тактической карте.
                    4. Треккинг мяча.
                    """
        )
        st.subheader("Как запустить?", divider="blue")
        st.markdown(
            """
                    **Есть два демонстрационных видеоролика, которые автоматически загружаются при запуске приложения, а также рекомендуемые настройки и гиперпараметры**
                    1. Загрузите видео для анализа, воспользовавшись кнопкой "Обзор файлов" в боковом меню.
                    2. Введите названия команд, соответствующие загруженному видео, в текстовые поля бокового меню.
                    3. Перейдите на вкладку "Гиперпараметры и обнаружение модели", настройте гиперпараметры и выберите параметры аннотации. (Рекомендуется использовать гиперпараметры по умолчанию)
                    4. Запустите детекцию!
                    5. Если была выбрана опция "сохранить выходные данные", то сохраненное видео можно найти в каталоге "выходные данные".
                    """
        )
        st.write("Version 0.0.1")

    with tab3:
        t2col1, t2col2 = st.columns([1, 1])
        with t2col1:
            df_field = st.empty()
            df_field.subheader("Данные детекции и трекинга:")
        with t2col2:

            st.write("Опции визуализации:")
            bcol21t, bcol22t = st.columns([1, 1])
            with bcol21t:
                show_k = st.toggle(label="Отобразить ключевые точки", value=False)
                show_p = st.toggle(label="Отобразить детекции игроков", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Отобразить трэки", value=True)
                show_b = st.toggle(label="Отобразить детекцию мяча", value=False)
            plot_hyperparser = {0: show_k, 1: show_pal, 2: show_b, 3: show_p}

            st.markdown("---")
            save_output = st.checkbox(label="Путь сохранения", value=False)
            if save_output:
                output_file_name = st.text_input(
                    label="Имя файла (Опционально)",
                    placeholder="Введите имя сохраняемого файла.",
                )
            else:
                output_file_name = None

        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            st.write("")
        with bcol2:
            # Инициализация сессионного состояния для кнопок
            if "start_pressed" not in st.session_state:
                st.session_state.start_pressed = False

            if "stop_pressed" not in st.session_state:
                st.session_state.stop_pressed = False

            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 1, 1, 1])

            with bcol22:
                ready = team1_name == "" or team2_name == ""
                start_detection = st.button(label="Начать детекцию", disabled=ready)
                if start_detection:
                    st.session_state.start_pressed = True
            with bcol23:
                stop_btn_state = not st.session_state.start_pressed
                stop_detection = st.button(
                    label="Остановить детекцию", disabled=stop_btn_state
                )
                if stop_detection:
                    st.session_state.stop_pressed = True
            with bcol21:
                flag = not (
                    st.session_state.start_pressed and st.session_state.stop_pressed
                )
                save_bt = st.button(label="Скачать файл", disabled=flag)

            with bcol24:
                st.write("")
        st.markdown("---")
    stframe = st.empty()
    cap = ffmpegcv.VideoCaptureNV(tempf.name)
    status = False

    if save_bt:
        data = get_data_from_db()

        # Конвертация данных в CSV
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Преобразование строки в байты
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        # Кнопка загрузки файла
        st.download_button(
            label="Скачать файл", data=csv_bytes, file_name="data.csv", mime="text/csv"
        )

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
        st.session_state.stop_pressed = True
        cap.release()


import contextlib

if __name__ == "__main__":
    with contextlib.suppress(SystemExit):
        main()
