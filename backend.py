from io import BytesIO
import time
import base64

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from Transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def img_to_base64_str(img):
    buffered = BytesIO()
    #img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def load_models(model_path):
    print("load model")
    with torch.no_grad():
        model = Transformer()
        state = torch.load(model_path)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        return model


gpu = -1

mapping_id_to_style = {0: "Hosoda", 1: "Hayao", 2: "Shinkai", 3: "Paprika"}

print(f"models loaded ...")


def transform(model, input, load_size=450, gpu=-1):
    models = {"Hosoda", "Hayao", "Shinkai", "Paprika"}
    models_dict = {0: "Hosoda", 1: "Hayao", 2: "Shinkai", 3: "Paprika"}

    if gpu > -1:
        model.cuda()
    else:
        model.float()

    input_image = Image.open(input).convert("RGB")
    h, w = input_image.size

    ratio = h * 1.0 / w

    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    input_image = -1 + 2 * input_image
    if gpu > -1:
        input_image = Variable(input_image).cuda()
    else:
        input_image = Variable(input_image).float()

    t0 = time.time()
    print("input shape", input_image.shape)
    with torch.no_grad():
        output_image = model(input_image)[0]
    print(f"inference time took {time.time() - t0} s")

    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    output_image = output_image.numpy()

    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    final_output_image = Image.fromarray(output_image)
    
    return final_output_image