from ultralytics import YOLO
from torch2trt import torch2trt
import torch
from torchreid.utils.feature_extractor import FeatureExtractor
import onnx

#model = YOLO('/container_dir/models/yolov8m_goalkeeper_1280.pt')
model_f = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='/container_dir/models/osnet_ain_x1_0_triplet_custom.pt', device='cuda')
torch_model = model_f.model
#model = model.eval().cuda()
x = torch.ones((30, 3, 256, 128)).cuda()
out = torch_model(x)

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "/container_dir/models/osnet_ain_x1_0_triplet_custom.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

onnx_model = onnx.load("/container_dir/models/osnet_ain_x1_0_triplet_custom.onnx")
onnx.checker.check_model(onnx_model)

#traced_model = torch.jit.trace(model.model, x)
model_trt = torch2trt(onnx_model, [x], fp16_mode=True)

torch.save(model_trt.state_dict(), '/container_dir/models/osnet_ain_x1_0_triplet_custom.engine')