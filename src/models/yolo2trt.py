from ultralytics import YOLO
import torch
from torch2trt import torch2trt
import tensorrt as trt
from torchreid.utils.feature_extractor import FeatureExtractor
import torch_tensorrt

model_path = '/container_dir/models/yolov8m_goalkeeper_1280.pt'

'''model_name = 'osnet_ain_x1_0'
loss = 'triplet'
use_gpu = torch.cuda.is_available()

model_f = FeatureExtractor(model_name=model_name, model_path='/container_dir/models/osnet_ain_x1_0_triplet_custom.pt', device='cuda')

    #torchreid.utils.load_pretrained_weights(model, '/container_dir/models/osnet_ain_x1_0_triplet_custom.pt')
model = model_f.model
model = model.eval().cuda()'''
model = YOLO(model_path)

model.export(format='engine', imgsz=640, half=True, dynamic=True)

'''inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 3, 256, 128],
        opt_shape=[23, 3, 256, 128],
        max_shape=[26, 3, 256, 128],
        dtype=torch.half,
    )]
trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions= torch.half)'''