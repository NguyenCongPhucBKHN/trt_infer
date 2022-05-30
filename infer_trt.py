import os
import sys
import time

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np




class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]





TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference(object):

    def load_engine(self, trt_runtime, engine_path):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    def allocate_buffers(self, engine):
   
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        # Current NMS implementation in TRT only supports DataType.FLOAT but
        # it may change in the future, which could brake this sample here
        # when using lower precision [e.g. NMS output would not be np.float32
        # anymore, even though this is assumed in binding_to_type]
        binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32}

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = binding_to_type[str(binding)]
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


    
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = None
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))
        if not os.path.exists(trt_engine_path):
            print("Incorrect path to engine")
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = self.load_engine(
                self.trt_runtime, trt_engine_path)
        
        self.inputs, self.outputs, self.bindings, self.stream = \
            self.allocate_buffers(self.trt_engine)

        self.context = self.trt_engine.create_execution_context()
        input_volume = trt.volume(3, 640, 640)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))


trt_model=TRTInference(trt_engine_path="/home/ivsr/CV_Group/phuc/tracking-by-sort-yolov5s-cpp/weight.engine")



