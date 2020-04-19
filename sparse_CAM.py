import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models, transforms
from model_MalConv import *
import pefile

# choose some test data (300?) to perform sparse-CAM


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device
device = get_device(True)
model = MalConv()
state = torch.load("./MalConv_5.pkl",map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# Images
# images, raw_images = load_images(image_paths)
# images = torch.stack(images).to(device)
first_n_byte = 2000000
# pe =  pefile.PE("putty.exe", fast_load=True)
result = {}
file_samples = ["processhacker-2.39-setup.exe"]
for sample in file_samples:
    
    pe =  pefile.PE(sample)
    with open(sample, 'rb') as f:
    # with open("putty.exe", 'rb') as f:
        bytes_array = np.array(bytearray(f.read()), dtype="uint8")
        # padding with zeroes such that all files will be of the same size
        if len(bytes_array) > first_n_byte:
            bytes_array = torch.Tensor(bytes_array[:first_n_byte])
        else:
            bytes_array = torch.Tensor(np.concatenate([bytes_array, np.zeros(first_n_byte - len(bytes_array),dtype='uint8')]))
    f.close()
    bytes_array = torch.stack((bytes_array,)).to(device)
    # print(bytes_array.shape)
    bytes_array.requires_grad = True
    dataloader = DataLoader(dataset=bytes_array, shuffle=False, batch_size=1)
    img = next(iter(dataloader)) # get the most likely prediction of the model 
    pred = model(img)
    pred.backward()

    gradients = model.get_activations_gradient()
    # print(gradients)
    count = 0
    base_addr = pe.OPTIONAL_HEADER.ImageBase
    for g in gradients[0]:
        values, indices = torch.max(g, 0)
        if values != 0:
            count += 1
    #         print(float(values), int(indices))
    #         print(hex(indices))
            sect = pe.get_section_by_offset((indices)*500)
            if sect == None:
                if (indices)*500 < pe.sections[0].PointerToRawData if pe.sections else min(len(pe.__data__), 0x1000):
                    target = "PE_header"
                else : 
                    target = "unknown_large"
            else:
                target = sect.Name
            try:
                result[target]+=1
            except KeyError:
                result[target]=1
print(result)
for i in pe.sections:
    
    print(i.Name)