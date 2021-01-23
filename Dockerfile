#
# docker build . --tag alae:latest
# docker run  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all  -it --rm --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  --mount type=bind,source="/home/jp/Documents/styleganWorkspace",target=/styleganWorkspace \ -v `pwd`:/var alae:latest
#
FROM nvcr.io/nvidia/pytorch:20.12-py3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt update -y
RUN apt-get upgrade -y
RUN apt-get install libgl1-mesa-glx -y

RUN pip install torch opencv-python torchvision tqdm scikit-image keras 
COPY . .
RUN pip install -r requirements.txt
RUN python training_artifacts/download_all.py
RUN python interactive_demo.py