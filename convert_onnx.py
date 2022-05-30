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
device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

path_to_weight_classify="/home/ivsr/CV_Group/phuc/ByteTrack/best_model_loss_SqueezeNet1_1_finetune.pth"
model_classify=SqueezeNet_BackBone()
model_classify=model_classify.to(device)
model_classify.load_state_dict(torch.load(path_to_weight_classify))
model_classify.eval()

resize=224

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

ONNX_FILE_PATH='./class2.onnx'
input=image_loader("/media/data/teamAI/phuc/data_classification/veri_test_d/SqueezeNet1_1/predict_save_img_type_color/bus_green_d/0065_c001_00064630_0.jpg")

torch.onnx.export(model_classify, input, ONNX_FILE_PATH, input_names=['input'],output_names=['output1', 'output2'], verbose=True, opset_version=11,export_params=True )

onnx_model = onnx.load(ONNX_FILE_PATH)
weights = onnx_model.graph.initializer
for w in weights:
    print(w.name)
# print(onnx.checker.check_model(onnx_model))



ort_session = onnxruntime.InferenceSession("/home/ivsr/CV_Group/phuc/TRT/trt_infer/class2.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
print(model_classify(input))
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
