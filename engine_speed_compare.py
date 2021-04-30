#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torchvision
from torchsummary import summary
import time
import pycuda.driver as cuda
import pycuda.autoinit

torch.manual_seed(0)


device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model=torch.load('./model/model_epoch_100.pth')["model"] 
model.to(device)
model.eval()

input_data = torch.randn(1, 1, 240, 240, dtype=torch.float32, device=device)  

output_data_x2, output_data_x4 = model(input_data)[0].cpu().detach().numpy(), model(input_data)[1].cpu().detach().numpy()

nRound = 10
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    model(input_data)
torch.cuda.synchronize()
time_pytorch = (time.time() - t0) / nRound
print('PyTorch time:', time_pytorch)

# input_names = ['input']
# output_names = ['output']
# torch.onnx.export(resnet50, input_data, 'resnet50.onnx', input_names=input_names, output_names=output_names, verbose=False, opset_version=11)
# torch.onnx.export(resnet50, input_data, 'resnet50.dynamic_shape.onnx', dynamic_axes={"input": [0, 2, 3]}, input_names=input_names, output_names=output_names, verbose=False, opset_version=11)

#继续运行python代码前，先运行如下命令
#trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt
#trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50_fp16.trt --fp16
#动态输入，仅供参考
#trtexec --verbose --onnx=resnet50.dynamic_shape.onnx --saveEngine=resnet50.dynamic_shape.trt --optShapes=input:1x3x1080x1920 --minShapes=input:1x3x1080x1920 --maxShapes=input:1x3x1080x1920

from trt_lite import TrtLite
import numpy as np
import os

class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor
    def get_pointer(self):
        return self.tensor.data_ptr()

for engine_file_path in ['./results/lapsrn_fp32.trt', './results/lapsrn_fp16.trt','./results/lapsrn_int8.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)
    
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (1, 1, 240, 240)}
    # batch_size=2
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    output_data_trt = np.zeros(io_info[1][2], dtype=np.float32)

    #利用PyTorch和PyCUDA的interop，保留数据始终在显存上
    cuda.memcpy_dtod(d_buffers[0], PyTorchTensorHolder(input_data), input_data.nelement() * input_data.element_size())
    #下面一行的作用跟上一行一样，不过它是把数据拷到cpu再拷回gpu，效率低。作为注释留在这里供参考
    #cuda.memcpy_htod(d_buffers[0], input_data.cpu().detach().numpy())
    trt.execute(d_buffers, i2shape)
    cuda.memcpy_dtoh(output_data_trt, d_buffers[1])

    cuda.Context.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute(d_buffers, i2shape)
    cuda.Context.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)

    print('Speedup:', time_pytorch / time_trt)
    print('Average diff percentage:', np.mean(np.abs(output_data_x2 - output_data_trt) / np.abs(output_data_x2)))