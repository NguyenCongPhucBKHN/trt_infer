import tensorrt as trt
import torch
from tabnanny import verbose
import torch2trt
import torch
import cv2
import sys
import random
import numpy as np
from classify.config import *
from classify.model import *
from os import listdir
from classify.transforms import ImageTransform
import sys
import onnx, onnxruntime

sys.path.append("/home/ivsr/CV_Group/phuc/Traffic_Color_Classification")

class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine Create execution context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names
    def trt_version(self):
        return trt.__version__
    def torch_dtype_from_trt(self, dtype):
        if dtype == trt.int8:
            return torch.int8
        elif trt.__version__>= '7.0' and dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError("%s is not supported by torch" % dtype)
    def torch_device_to_trt(self,device):
        if device.type == torch.device("cuda").type:
            return trt.TensorLocation.DEVICE
        elif device.type == torch.device("cpu").type:
            return trt.TensorLocation.HOST
        else:
            return TypeError("%s is not supported by tensorrt" % device)
    
    def torch_device_from_trt(self, device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device("cuda")
        elif device == trt.TensorLocation.HOST:
            return torch.device("cpu")
        else:
            return TypeError("%s is not supported by torch" % device)


    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            #  Set up shape 
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()
        
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = self.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = self.torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
loader = transforms.Compose([transforms.Resize((resize, resize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])
def image_loader(image_path):
    """load image, returns cuda tensor"""
    image = Image.open(image_path)
    image = image.convert('RGB')

    image = loader(image).float()
    # print(image.shape)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda() 
logger = trt.Logger(trt.Logger.INFO)
with open("./class2.trt", "rb") as f, trt.Runtime(logger) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())
model_all_names=[]
for idx in range(engine.num_bindings):
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    op_type = engine.get_binding_dtype(idx)
    model_all_names.append(name)
    shape = engine.get_binding_shape(idx)

    print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)

trt_model=TRTModule(engine=engine, input_names=["input"], output_names=["output1", "output2"])
input=image_loader("/media/data/teamAI/phuc/data_classification/veri_test_d/SqueezeNet1_1/predict_save_img_type_color/bus_green_d/0065_c001_00064630_0.jpg")
result_trt=trt_model(input)
print(result_trt)