import torch
import torch_tensorrt
from torchreid.utils.feature_extractor import FeatureExtractor

model_reid = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="models/osnet_ain_x1_0_triplet_custom.pt",
    device="cuda",
)
model = model_reid.model

model.eval()
model.cuda()

min_batch_size = 1
max_batch_size = 23
example_input = torch.randn(max_batch_size, 3, 256, 128).cuda()

trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        example_input,
        torch_tensorrt.Input(
            min_shape=(min_batch_size, 3, 256, 128),
            opt_shape=(max_batch_size, 3, 256, 128),
            max_shape=(max_batch_size, 3, 256, 128),
            dtype=torch.half,
        ),
    ],
    enable_precision={torch.half},
)

test_batch = torch.randn(10, 3, 256, 128)

with torch.no_grad():
    output = trt_model(test_batch)

print(output)
