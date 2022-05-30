from distutils.command.build import build
from yaml import parse
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch
import sys
import torch2trt
import torch
import cv2

import random
import numpy as np
from classify.config import *
from classify.model import *
from os import listdir
from classify.transforms import ImageTransform
import onnxruntime
import numpy as np
import onnx
sys.path.append("/home/ivsr/CV_Group/phuc/Traffic_Color_Classification")

# TRT_LOGGER=trt.Logger()
# loader = transforms.Compose([transforms.Resize((resize, resize)),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize(mean, std)])

# def image_loader(image_path):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_path)
#     image = image.convert('RGB')

#     image = loader(image).float()
#     # print(image.shape)
#     image = Variable(image, requires_grad=False)
#     image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
#     return image.cuda() 

# def build_engine(onnx_file_path):
#     onnx_file_path="/home/ivsr/CV_Group/phuc/TRT/trt_infer/classification.onnx"
#     builder=trt.Builder(TRT_LOGGER)
#     network=builder.create_network()
#     parse=trt.OnnxParser(network, TRT_LOGGER)
#     with open(onnx_file_path, 'rb') as model:
#         print("Begin ONNX file parsing ")
#         parse.parse(model.read())
#     print("Complete parsing of ONNX file")



#     # builder.max_workspace_size=1<<30
#     builder.max_batch_size=1

#     if builder.platform_has_fast_fp16:
#         builder.fp16_mode=True

#         print("Buidling an engine ...")
#         engine=builder.build_cuda_engine(network)
#         context=engine.create_execution_context()
#         print("Complete creating Engine")
#         print(engine)
#         return engine, context


# def main(ONNX_FILE_PATH):
#     engine, context= build_engine(ONNX_FILE_PATH)
#     for binding in engine:
#         if engine.binding_is_input(binding):
#             input_shape= engine.get_binding_shape(binding)
#             input_size=trt.volume(input_shape)*engine.max_batch_size*np.dtype(np.float32).itemsize
#             device_input=cuda.mem_alloc(input_size)
#         else:
#             output_shape=engine.get_binding_shape(binding)
#             host_output=cuda.pagelocked_empty(trt.volume(output_shape)*engine.max_batch_size, dtype=np.float32)
#             device_output=cuda.mem_alloc(host_output.nbytes)
#     stream=cuda.Stream()
#     host_input=np.array(image_loader("/media/data/teamAI/phuc/data_classification/veri_test_d/SqueezeNet1_1/predict_save_img_type_color/bus_green_d/0065_c001_00064630_0.jpg").numpy(),
#     dtype=np.float32, order='C')
#     cuda.memcpy_htod_async(device_input, host_input, stream)

#     context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle )
#     cuda.memcpy_dtoh_async(host_output, device_output, stream)
#     stream.synchronize()
#     output_data=torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])

# # build_engine("/home/ivsr/CV_Group/phuc/TRT/trt_infer/classification.onnx")
# # main("/home/ivsr/CV_Group/phuc/TRT/trt_infer/classification.onnx")







# onnx_file_path="/home/ivsr/CV_Group/phuc/TRT/trt_infer/class2.onnx"
# builder=trt.Builder(TRT_LOGGER)
# network=builder.create_network()
# parse=trt.OnnxParser(network, TRT_LOGGER)
# with open(onnx_file_path, 'rb') as model:
#     print("Begin ONNX file parsing ")
#     if(not parse.parse(model.read())):
#         print("Done")


# print(network.num_layers)

# # onnx_file_path="/home/ivsr/CV_Group/phuc/TRT/trt_infer/class2.onnx"
# # builder=trt.Builder(TRT_LOGGER)
# # network=builder.create_network()
# # print(network)
# # parse=trt.OnnxParser(network, TRT_LOGGER)
# # with open(onnx_file_path, 'rb') as model:
# #     print("Begin ONNX file parsing ")
# #     print(parse.parse(model.read()))

# # print("Complete parsing of ONNX file")
# # builder.max_batch_size=1

# # # if builder.platform_has_fast_fp16:
# # # builder.fp16_mode=True

# # print("Buidling an engine ...")



# # # engine=builder.build_cuda_engine(network)
# # # context=engine.create_execution_context()
# # # print("Complete creating Engine")
# # # print(engine)

# # # engine=builder.build_cuda_engine(network)
# # # context=engine.create_execution_context()
# # # print(engine)

# # profile=builder.create_optimization_profile()

# # config=builder.create_builder_config()

# # config.add_optimization_profile(profile)
# # network=builder.create_network()
# # engine=builder.build_serialized_network(network, config)
# # print(engine)


# # context=engine.create_exection_context()







    
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#         with open("./class1.onnx", 'rb') as model:
#             config = builder.create_builder_config()
#             config.max_workspace_size = 10000
#             if not parser.parse(model.read()):
#                 for error in range(parser.num_errors):
#                     print(parser.get_error(error))
#         # engine = builder.build_cuda_engine(network)
#         plan = builder.build_serialized_network(network, config)
#         with trt.Runtime(TRT_LOGGER) as runtime:
#             engine = runtime.deserialize_cuda_engine(plan)



import tensorrt as trt

def ONNX_build_engine(onnx_file_path, engine_file_path):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = 16
        # builder.max_workspace_size = 1 << 20

        # print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            print(parser.parse(model.read()))
            
        # print('Completed parsing of ONNX file')
        # print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        
        
        # print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            config = builder.create_builder_config()
            plan = builder.build_serialized_network(network, config)
            
            with trt.Runtime(G_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(plan)
                print(engine.serialize())

                f.write(engine.serialize())
        print("Donde")
        # engine = builder.build_cuda_engine(network)
        return engine
def main(engine):
    # engine, context= build_engine(ONNX_FILE_PATH)
    for binding in engine:
        print(binding)
        if engine.binding_is_input(binding):
            input_shape= engine.get_binding_shape(binding)
            input_size=trt.volume(input_shape)*engine.max_batch_size*np.dtype(np.float32).itemsize
            device_input=cuda.mem_alloc(input_size)
            print("IF: ", input_size)
        else:
            output_shape=engine.get_binding_shape(binding)
            host_output=cuda.pagelocked_empty(trt.volume(output_shape)*engine.max_batch_size, dtype=np.float32)
            device_output=cuda.mem_alloc(host_output.nbytes)
            print("ELSE: ", input_size)
    # stream=cuda.Stream()
    # host_input=np.array(image_loader("/media/data/teamAI/phuc/data_classification/veri_test_d/SqueezeNet1_1/predict_save_img_type_color/bus_green_d/0065_c001_00064630_0.jpg").numpy(),
    # dtype=np.float32, order='C')
    # cuda.memcpy_htod_async(device_input, host_input, stream)

    # context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle )
    # cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # stream.synchronize()
    # output_data=torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])

# # build_engine("/home/ivsr/CV_Group/phuc/TRT/trt_infer/classification.onnx")
# # main("/home/ivsr/CV_Group/phuc/TRT/trt_infer/classification.onnx")

engine=ONNX_build_engine('./class1.onnx', './class3.trt')
main(engine)


