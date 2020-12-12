import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import csv
import copy
import os


from ESPCN_model import ESPCN
from data_loader import ImageDataset
from utils import PSNR, write_csv

def train(scale, device):
    
    # initialize model
    model = ESPCN(upscale_factor=scale).to(device)

    # MSE loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[15, 80], gamma=0.1)

    # load the data
    train_dataset = ImageDataset('data/processed/train', upscale_factor=scale,
                                  input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())

    val_dataset = ImageDataset('data/processed/val', upscale_factor=scale, 
                                input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=4)

    
    # Train Model
    epoch = 100
    train_loss = []
    train_psnr = []
    
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    for epoch in range(1, epoch+1):
        # Train Model
        model.train()
        loss = 0
        losses = []
        print("Starting epoch number " + str(epoch))

        for i, (images, labels) in enumerate(train_loader):
            # images shape torch.Size([64, 1, 85, 85])
            # labels shape torch.Size([64, 1, 255, 255])
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad()
            out_images = model(images)

            loss = criterion(out_images, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        
        loss = torch.stack(losses).mean().item()
        train_loss.append(loss)
        print("Loss for Training on Epoch " +str(epoch) + " is "+ "{:.6f}".format(loss))
        
        # save model
        save_dir = 'saved_models/UPSCALE_X' + str(scale) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_{}.pth'.format(epoch)))

        # Evaluate Model
        model.eval()
        psnr = 0
        psnrs = []
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device) 
            
            with torch.no_grad():
                out_images = model(images)
            
            psnr = PSNR(out_images, labels)
            psnrs.append(psnr)

        psnr = torch.stack(psnrs).mean().item()
        train_psnr.append(psnr)
        print('Eval PSNR: {:.2f}\n'.format(psnr))

        if psnr > best_psnr:
            best_epoch = epoch
            best_psnr = psnr
            best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(save_dir, 'best.pth'))

    # write PSNR to CSV file
    csv_name = 'results/Eval_PSNR_X' + str(scale) + '.csv'
    write_csv(csv_name, train_psnr, scale)

    # write losses to CSV file
    file_path = 'results/train_loss_X' + str(scale) + '.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    solution_rows = [('epoch', 'train_loss', 'Upscale')] + [(i, y, scale) for (i, y) in enumerate(train_loss)]
    with open(file_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)



if __name__ == "__main__":
    # check GPU availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Upscale Factor
    
    train(8, device)


