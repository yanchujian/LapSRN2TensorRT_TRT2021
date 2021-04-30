import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import argparse
from torch.autograd import Variable
from torchvision.transforms import ToTensor

TRT_LOGGER = trt.Logger()
def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
#和上面没什么区别
def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def do_inference(engine, batch_size, input, output_shape):
    print(input.shape)
    print(output_shape)
    context = engine.create_execution_context()
    # print(context) #数组元素不为空,随机产生的数据
    output = np.empty(output_shape[0], dtype=np.float32) #output_shape 960*960
    output2 = np.empty(output_shape[1], dtype=np.float32)
    print(output.size)

    # 分配内存
    d_input = cuda.mem_alloc(1 * input.size * input.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    d_output2=cuda.mem_alloc(1 * output2.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output),int(d_output2)]

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, input, stream)

    start = time.time()
    # 执行模型
    context.execute_async(batch_size, bindings, stream.handle, None)
    # 将预测结果从从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    cuda.memcpy_dtoh_async(output2, d_output2, stream)
    end = time.time()

    # 线程同步
    stream.synchronize()
    fp16_time=end-start

    #
    return output,output2,fp16_time
    # print("\nTensorRT {} test:".format(engine_path.split('/')[-1].split('.')[0]))
    # print("output:", output)
    # print("output2:", output2)
    # print("time cost:", end - start)

def get_shape(engine):

    output_shape=[]
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
        else:
            output_shape.append(list(engine.get_binding_shape(binding)))
    return input_shape, output_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "TensorRT do inference")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--img_path", type=str, default='test_image/baby_x2.png', help='cache_file')
    parser.add_argument("--engine_file_path", type=str, default='lapsrn_fp32.trt', help='engine_file_path')
    args = parser.parse_args()

    engine_path = args.engine_file_path
    # engine = loadEngine2TensorRT(engine_path) #反序列化
    engine=get_engine(engine_path)
    # print(engine)
    img = Image.open(args.img_path)
    img_L=img.convert("YCbCr")
    y, cb, cr = img_L.split()
    

    input_shape, output_shape = get_shape(engine)
    transform = transforms.Compose([
        transforms.Resize([input_shape[2], input_shape[3]]),  # [h,w]
        transforms.ToTensor()
        ])
    # img = transform(img).unsqueeze(0) #升维
    img=transform(y).unsqueeze(0)
    img = img.numpy()
    # img= Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])

    
    print(input_shape,output_shape)
    output,output2,fp16_time=do_inference(engine, args.batch_size, img, output_shape) #输出需要转换cpu吗
    output=output.squeeze(0)
    # out_img_y = output.data[0].numpy()
    out_img_y=output
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    w, h = out_img_y.size
    out_img_r = cr.resize((w, h), Image.ANTIALIAS)
    out_img_b = cb.resize((w, h), Image.ANTIALIAS)

    out_img = Image.merge('YCbCr', [out_img_y, out_img_b, out_img_r]).convert('RGB')
    out_img.save('test_result/baby_fp32.jpg')
    print(fp16_time)
    
