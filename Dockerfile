FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN apt-get update && apt-get install -y ca-certificates
RUN apt update && apt upgrade -y
RUN apt install -y ffmpeg 

RUN python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt
#RUN pip uninstall opencv-python
RUN pip install opencv-python==4.8.0.74

WORKDIR /container_dir

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install && \
    cmake -B build . && cmake --build build --target install && ldconfig && \
    cd .. 

RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git && \
    cd deep-person-reid && \
    pip install -r requirements.txt && \
    python setup.py develop

#RUN cd Deep-EIoU/Deep-EIoU
#ENTRYPOINT ["streamlit", "run", "tools/main.py"]
#COPY entrypoint.sh /entrypoint.sh
#RUN chmod +x /entrypoint.sh
#ENTRYPOINT ["/entrypoint.sh"]