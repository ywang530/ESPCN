import os
from os import listdir
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def generate_dataset(root_dir ,data_type, upscale_factor, crop):
    images_name = [x for x in listdir(root_dir + '/' + data_type)]
    crop_size = crop - (crop % upscale_factor)

    # low resolution images generation
    lr_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                   transforms.Resize(crop_size//upscale_factor, interpolation=Image.BICUBIC)])
    
    # high resolution(original) images generation
    hr_transform = transforms.Compose([transforms.CenterCrop(crop_size)])
    
    root = 'data/processed/' + data_type
    if not os.path.exists(root):
        os.makedirs(root)

    path = root + '/UPSCALE_X' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
    
    image_path = path + '/data'
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    target_path = path + '/target'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
            + str(upscale_factor) + ' from VOC2012'):
        image = Image.open(root_dir + '/' + data_type + '/' + image_name)
        target = image.copy()
        image = lr_transform(image)
        target = hr_transform(target)

        image.save(image_path + '/' + image_name)
        target.save(target_path + '/' + image_name)


if __name__ == "__main__":
    for i in range(2,6):
        generate_dataset(root_dir='data/original', data_type='train', upscale_factor=i, crop = 256)
        generate_dataset(root_dir='data/original', data_type='val', upscale_factor=i, crop = 256)