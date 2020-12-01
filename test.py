import os
from os import listdir
from PIL import Image
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import numpy as np
from torchvision import transforms
import csv


from ESPCN_model import ESPCN
from utils import PSNR, write_csv

def test(data_path, scale, device, results_path, csv_path, label):
    images_name = [x for x in listdir(data_path)]

    model = ESPCN(upscale_factor=scale).to(device)
    model.load_state_dict(torch.load('saved_models/' + 'UPSCALE_X' + str(scale)+ '/' + 'best.pth'))

    ESPCN_PSNR = []
    BICUBIC_PSNR = []

    output_path = results_path + 'UPSCALE_X' + str(scale) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bicubic_path = results_path + 'BICUBIC_X' + str(scale) + '/'
    if not os.path.exists(bicubic_path):
        os.makedirs(bicubic_path)

    lr_path = results_path + 'LR_X' + str(scale) + '/'
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)
   
    for image_name in tqdm(images_name, desc=('convert LR images to HR images - ' + label)):
        image = Image.open(data_path + image_name)

        # produce LR and HR image 
        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale

        HR = image.resize((image_width, image_height), resample=Image.BICUBIC)
        LR = HR.resize((HR.width//scale, HR.height//scale), resample=Image.BICUBIC)
        LR.save(lr_path + image_name)

        # produce # reference bicubic interpolation image
        bicubic = LR.resize((LR.width*scale, LR.height*scale), resample=Image.BICUBIC)
        bicubic.save(bicubic_path + image_name)

        # produce SR image
        LR_Y, LR_Cb, LR_Cr = LR.convert('YCbCr').split()
        input_image = transforms.ToTensor()(LR_Y).view(1, -1, LR_Y.size[1], LR_Y.size[0]).to(device)
        
        output_image = model(input_image)
        output_Y = output_image.data[0].cpu().numpy()
        output_Y *= 255
        output_Y = output_Y.clip(0,255)
        output_Y = Image.fromarray(np.uint8(output_Y[0]), mode='L')

        output_Cb = LR_Cb.resize(output_Y.size, Image.BICUBIC)
        output_Cr = LR_Cr.resize(output_Y.size, Image.BICUBIC)

        SR = Image.merge('YCbCr', [output_Y, output_Cb, output_Cr]).convert('RGB')
        SR.save(output_path + image_name)

        # calculate PSNR
        HR_Y, _, _ = HR.convert('YCbCr').split()
        HR_Y = transforms.ToTensor()(HR_Y).to(device)
        bicubic_Y, _, _ = bicubic.convert('YCbCr').split()
        bicubic_Y = transforms.ToTensor()(bicubic_Y).to(device)

        SR_Y = output_image

        ESPCN_PSNR.append(PSNR(SR_Y, HR_Y).item())
        BICUBIC_PSNR.append(PSNR(bicubic_Y, HR_Y).item())
   
    # write to CSV
    file_path = csv_path + '_X' + str(scale) + '.csv'
    solution_rows = [('Image', 'ESPCN PSNR', 'BICUBIC PSNR','Upscale')] + [(y, ESPCN_PSNR[i], BICUBIC_PSNR[i], scale) for (i, y) in enumerate(images_name)]
    with open(file_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # test on Set5, Set14, and Celeba Dataset

    Sets = ['Set5', 'Set14', 'celeba']

    for name in Sets:
        for scale in range (2,6):
            data_path = 'data/test/' + name + '/'

            results_path = 'results/' + name + '/' 
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv_path = name + '_ESPCN_BICUBIC_Comparison' 

            test(data_path, scale, device, results_path, csv_path, name)
    