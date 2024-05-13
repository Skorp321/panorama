#start container vith cuda, torch and tansorrt
docker run --net=host --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/skorp321/Projects/panorama:/container_dir tensorrt08