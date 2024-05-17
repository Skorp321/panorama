from torch2trt import torch2trt
import torch
from torchreid.utils.feature_extractor import FeatureExtractor
import onnx

#model = YOLO('/container_dir/models/yolov8m_goalkeeper_1280.pt')
model_f = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='/container_dir/models/osnet_ain_x1_0_triplet_custom.pt', device='cuda')
torch_model = model_f.model
#model = model.eval().cuda()
input = torch.ones(4, 3, 224, 224).cuda()
out = torch_model(input)
print(len(out))

model_trt = torch2trt(torch_model, [input], max_batch_size=26)

input = torch.ones(4, 3, 224, 224).cuda()
res = model_trt(input)
print(len(res))

input = torch.ones(24, 3, 224, 224).cuda()
res = model_trt(input)
print(len(res))

#torch.save(model_trt.state_dict(), '/container_dir/models/osnet_ain_x1_0_triplet_custom.engine')