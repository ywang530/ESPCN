import csv
import torch

# function to write PSNR to csv files
def write_csv(file_path, y_list, upscale_factor):
    solution_rows = [('epoch', 'Eval PSNR', 'Upscale')] + [(i, y, upscale_factor) for (i, y) in enumerate(y_list)]

    with open(file_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)


# function to calculate Peak signal-to-noise ratio
def PSNR(image, target):
    return 10 * torch.log10(1 / torch.mean((image-target)**2))