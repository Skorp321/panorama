import contextlib
import os
import tempfile
import numpy as np

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detectionApp import detect
import ffmpegcv
import multiprocessing


def main():

    st.set_page_config(page_title="AI Powered Web Application for Football Tactical Analysis", layout="wide", initial_sidebar_state="expanded")
    # Добавление пользовательских стилей
    st.markdown("""
    <style>
    body {
        background-color: #333;
        color: #ddd;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Детекция футбольных игроков с разбиением на команды и проекцией на тактическую карту.")
    st.subheader(":red[Хорошо работает только с панорамным видео!]")

    st.sidebar.title("Основные настройки")
    demo_selected = st.sidebar.radio(label="Выбирите демо видео:", options=["Demo 1", "Demo 2"], horizontal=True)

    ## Sidebar Setup
    st.sidebar.markdown('---')
    st.sidebar.subheader("Загрузка видео")
    input_vide_file = st.sidebar.file_uploader('Загрузите видео файл', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    demo_vid_paths={
        "Demo 1":'/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4',
        "Demo 2":'/container_dir/data/Swiss_vs_slovakia-Panorama.mp4'
    }
    demo_vid_path = demo_vid_paths[demo_selected]
    demo_team_info = {
        "Demo 1":{"team1_name":"France",
                  "team2_name":"Switzerland",
                  "team1_p_color":'#1E2530',
                  "team1_gk_color":'#F5FD15',
                  "team2_p_color":'#FBFCFA',
                  "team2_gk_color":'#B1FCC4',
                  },
        "Demo 2":{"team1_name":"Chelsea",
                  "team2_name":"Manchester City",
                  "team1_p_color":'#29478A',
                  "team1_gk_color":'#DC6258',
                  "team2_p_color":'#90C8FF',
                  "team2_gk_color":'#BCC703',
                  }
    }
    selected_team_info = demo_team_info[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
    else:
        tempf.write(input_vide_file.read())
    demo_vid = open(tempf.name, 'rb')
    demo_bytes = demo_vid.read()

    st.sidebar.text('Демо видео:')
    st.sidebar.video(demo_bytes)
    # Load the YOLOv8 players detection model
    model_players = YOLO("../../models/yolov8m_goalkeeper_1280.pt")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("../../models/Yolo8M Field Keypoints/weights/best.pt")


    st.sidebar.markdown('---')
    st.sidebar.subheader("Названия команд")
    team1_name = st.sidebar.text_input(label='Имя первой команды', value=selected_team_info["team1_name"])
    team2_name = st.sidebar.text_input(label='Имя второй команды', value=selected_team_info["team2_name"])
    st.sidebar.markdown('---')

    ## Page Setup
    tab1, tab3 = st.tabs(["Как запустить?", "Настройки детекции"])
    with tab1:
        st.header(':blue[Welcome!]')
        st.subheader('Основные возможности приложения:', divider='blue')
        st.markdown("""
                    1. Детекция игроков, судей и мяча.
                    2. Разбиение игроков на команды.
                    3. Отображение положения игроков и мяча на тактической карте.
                    4. Треккинг мяча.
                    """)
        st.subheader('Как запустить?', divider='blue')
        st.markdown("""
                    **Есть два демонстрационных видеоролика, которые автоматически загружаются при запуске приложения, а также рекомендуемые настройки и гиперпараметры**
                    1. Загрузите видео для анализа, воспользовавшись кнопкой "Обзор файлов" в боковом меню.
                    2. Введите названия команд, соответствующие загруженному видео, в текстовые поля бокового меню.
                    3. Перейдите на вкладку "Гиперпараметры и обнаружение модели", настройте гиперпараметры и выберите параметры аннотации. (Рекомендуется использовать гиперпараметры по умолчанию)
                    4. Запустите детекцию!
                    5. Если была выбрана опция "сохранить выходные данные", то сохраненное видео можно найти в каталоге "выходные данные".
                    """)
        st.write("Version 0.0.1")

    with tab3:
        t2col1, t2col2 = st.columns([1,1])
        with t2col1:
            player_model_conf_thresh = st.slider('PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.6)
            keypoints_model_conf_thresh = st.slider('Field Keypoints PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.7)
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement RMSE Tolerance (pixels)', min_value=-1, max_value=100, value=7,
                                                        help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                        in current and previous detections. It is used to determine wether to update homography matrix or not. ")
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
                                    help="How many colors to extract form detected players bounding-boxes? It is used for team prediction.")
            st.markdown("---")
            save_output = st.checkbox(label='Save output', value=False)
            if save_output:
                output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
            else:
                output_file_name = None
        st.markdown("---")

        bcol1, bcol2 = st.columns([1,1])
        with bcol1:
            nbr_frames_no_ball_thresh = st.number_input("Ball track reset threshold (frames)", min_value=1, max_value=10000,
                                                        value=30, help="After how many frames with no ball detection, should the track be reset?")
            ball_track_dist_thresh = st.number_input("Ball track distance threshold (pixels)", min_value=1, max_value=1280,
                                                        value=100, help="Maximum allowed distance between two consecutive balls detection to keep the current track.")
            max_track_length = st.number_input("Maximum ball track length (Nbr. detections)", min_value=1, max_value=1000,
                                                        value=35, help="Maximum total number of ball detections to keep in tracking history")
            ball_track_hyperparams = {
                0: nbr_frames_no_ball_thresh,
                1: ball_track_dist_thresh,
                2: max_track_length
            }
        with bcol2:
            st.write("Опции визуализации:")
            bcol21t, bcol22t = st.columns([1,1])
            with bcol21t:
                show_k = st.toggle(label="Отобразить ключевые точки", value=False)
                show_p = st.toggle(label="Отобразить детекции игроков", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Отобразить трэки", value=True)
                show_b = st.toggle(label="Отобразить детекцию мяча", value=False)
            plot_hyperparser = {
                0: show_k,
                1: show_pal,
                2: show_b,
                3: show_p
            }
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5,1,1,1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = team1_name == '' or team2_name == ''
                start_detection = st.button(label='Start Detection', disabled=ready)
            with bcol23:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
            with bcol24:
                st.write('')


    stframe = st.empty()
    cap = ffmpegcv.VideoCaptureNV(tempf.name)
    status = False

    if start_detection and not stop_detection:
        st.toast('Detection Started!')
        parent_dir = "/container_dir/Deep-EIoU/Deep-EIoU"

        # Изменение текущего рабочего каталога
        os.chdir(parent_dir)
        status = detect(cap, stframe, output_file_name, save_output, plot_hyperparser)

    else:
        with contextlib.suppress(Exception):
            # Release the video capture object and close the display window
            cap.release()
    if status:
        st.toast('Detection Completed!')
        cap.release()
        status.terminate()


import contextlib
if __name__=='__main__':
    with contextlib.suppress(SystemExit):
        main()