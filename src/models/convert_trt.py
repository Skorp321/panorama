import json
import time

import numpy as np
import torchreid
import torch
import torch.onnx
import tensorrt as trt
import torch_tensorrt
from collections import OrderedDict, namedtuple
from torchreid.utils.feature_extractor import FeatureExtractor



model_name = 'osnet_ain_x1_0'
loss = 'triplet'
use_gpu = torch.cuda.is_available()

def benchmark(model, input_shape=(26, 3, 256, 128), dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

def main():
    import torch
    '''model = torchreid.models.build_model(
        name=model_name,
        num_classes=946,
        loss=loss,
        pretrained=False,       
    )'''

    model_f = FeatureExtractor(model_name=model_name, model_path='/container_dir/models/osnet_ain_x1_0_triplet_custom.pt', device='cuda')

    #torchreid.utils.load_pretrained_weights(model, '/container_dir/models/osnet_ain_x1_0_triplet_custom.pt')
    model = model_f.model
    model = model.eval().cuda()  # torch module needs to be in eval (not training) mode

    #input = torch.ones((4, 3, 224, 224)).cuda()
    inputs = torch.randn((26, 3, 256, 128)).cuda()
    batch_size = 26
    
    print('Torch model:')
    benchmark(model)
    
    with torch.no_grad():
        jit_model = torch.jit.trace(model, inputs)
    
    print('TorchScript model:')    
    benchmark(jit_model)
    '''
    trt_model_fp32 = torch_tensorrt.compile(jit_model, 
                                            ir='torchscript',
                                            inputs=[torch_tensorrt.Input((batch_size, 3, 256, 128), dtype=torch.float)],
                                            enabled_precisions={torch.float},
                                            truncate_long_and_double=True)
    
    print('tensorrt model 32:')
    benchmark(trt_model_fp32)'''
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    trt_model_fp16 = torch_tensorrt.compile(jit_model,
                                        inputs = torch_tensorrt.Input(min_shape=(1, 256, 128, 3),
                                                                      opt_shape=(23, 256, 128, 3),
                                                                      max_shape=(26, 256, 128, 3),
                                                                      format=torch.channel_last,
                                                                      dtype=torch.half))
    

        
    print('tensorrt model 16:')
    benchmark(trt_model_fp16, dtype='fp16')     
    torch.jit.save(trt_model_fp16, 'models/osnet_ain_x1_0_triplet_custom.engine')
    #trt_model_fp16._save_to_state_dict('models/osnet_x1_01.engine')
    '''with open('models/osnet_x1_02.engine', 'wb') as f:
        f.write(trt_model_fp16)'''
    '''
    w = 'models/osnet_x1_0.pt'
    
    logger = trt.Logger(trt.Logger.INFO)
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        
        meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
        metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
        model = runtime.deserialize_cuda_engine(f.read())  # read engine
    context = model.create_execution_context()
    bindings = OrderedDict()
    output_names = []
    fp16 = False  # default updated below
    dynamic = False
    for i in range(model.num_bindings):
        name = model.get_binding_name(i)
        dtype = trt.nptype(model.get_binding_dtype(i))
        if model.binding_is_input(i):
            if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                dynamic = True
                context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
            if dtype == np.float16:
                fp16 = True
        else:  # output
            output_names.append(name)
        shape = tuple(context.get_binding_shape(i))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
    
    ''' 
if __name__ == "__main__":
    main()
