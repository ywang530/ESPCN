# Efficient Sub-Pixel Convolutional Neural Network (ESPCN)
 A PyTorch reimplementation of ESPCN based on paper [***Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network***](https://arxiv.org/abs/1609.05158)

## Requirements
- Pytorch
- Anaconda

## Datasets
The dataset I used to train and validate the model is VOC2012 (Pascal VOC challenge). The dataset contains 17,125 images of 20 objects: person, bird, cat, cow, dog, horse, sheep, airplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv/monitor. 

75% of the images, 12,844, are partitioned to be training set, and 25% of the images, 4,281, are partitioned to be validation set. 

To generate the LR input images, I downsampled the imaged corresponding to different upscale factors. For this project, I tested upscale factor from 2 to 8. The model is tested on Set 5 (5 images), Set 14 (14 images), and CelebA (first 30 images) datasets.

## Implementation
The model used to conduct this project contains 3 layers: 
- 𝐹𝑖𝑟𝑠𝑡 𝐻𝑖𝑑𝑑𝑒𝑛 𝐿𝑎𝑦𝑒𝑟: 𝐶ℎ𝑎𝑛𝑛𝑒𝑙 𝐼𝑛=1,𝐶ℎ𝑎𝑛𝑛𝑒𝑙 𝑂𝑢𝑡=64,𝐾𝑒𝑟𝑛𝑒𝑙 𝑠𝑖𝑧𝑒=5 
- 𝑆𝑒𝑐𝑜𝑛𝑑 𝐻𝑖𝑑𝑑𝑒𝑛 𝐿𝑎𝑦𝑒𝑟: 𝐶ℎ𝑎𝑛𝑛𝑒𝑙 𝐼𝑛=64,𝐶ℎ𝑎𝑛𝑛𝑒𝑙 𝑂𝑢𝑡=32,𝐾𝑒𝑟𝑛𝑒𝑙 𝑠𝑖𝑧𝑒=3 
- 𝑇ℎ𝑖𝑟𝑑 𝐻𝑖𝑑𝑑𝑒𝑛 𝐿𝑎𝑦𝑒𝑟: 𝐶ℎ𝑎𝑛𝑛𝑒𝑙 𝐼𝑛=32,𝐶ℎ𝑎𝑛𝑛𝑒𝑙 𝑂𝑢𝑡 =𝑈𝑝𝑠𝑐𝑎𝑙𝑒 𝐹𝑎𝑐𝑡𝑜𝑟2, 𝐾𝑒𝑟𝑛𝑒𝑙 𝑠𝑖𝑧𝑒=3

The reason why the input channel is 1 is that the color space used here is YCbCr and only Y channel is considered and processed as human eyes are more sensitive to luminance.
First two Hidden Layers are followed by Tanh activation function, while the last one is followed by Pixel Shuffle (sub-pixel convolution) which rearranges the elements in of shape (∗, 𝐶×𝑟2, 𝐻, 𝑊) to shape (∗, 𝐶, 𝐻× 𝑟, 𝑊× 𝑟).

The Loss Function used is pixel-wise mean squared error (MSE) and the Optimizer used is Adam optimizer. The initial learning rate is set to 1𝑒−2, and will decrease to 1𝑒−3 after epochs 15 then decrease to 1𝑒−4 after epochs 80. For the experiment, the total Epoch is set to be 100.

To evaluate the performance of the model, I used PSNR in dB as a metric to compare with commonly used bicubic interpolation:

<img src="https://latex.codecogs.com/svg.latex?PSNR&space;=&space;10\times&space;\log_{10}\frac{1}{MSE}" title="PSNR = 10\times \log_{10}\frac{1}{MSE}" />

##  Results Comparison 
### Upscale x2
|Dataset| Input        | ESPCN         |   Bicubic    | Original |
| ------------- | ------------- | ------------- |------------- | ------------- |
|Set 14|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/LR_X2/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/UPSCALE_X2/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/BICUBIC_X2/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set14/comic.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/LR_X2/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/UPSCALE_X2/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/BICUBIC_X2/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set5/butterfly.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/LR_X2/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/UPSCALE_X2/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/BICUBIC_X2/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/celeba/000010.png)|

### Upscale x3
|Dataset| Input        | ESPCN         |   Bicubic    | Original |
| ------------- | ------------- | ------------- |------------- | ------------- |
|Set 14|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/LR_X3/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/UPSCALE_X3/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/BICUBIC_X3/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set14/comic.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/LR_X3/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/UPSCALE_X3/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/BICUBIC_X3/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set5/butterfly.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/LR_X3/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/UPSCALE_X3/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/BICUBIC_X3/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/celeba/000010.png)|

### Upscale x4
|Dataset| Input        | ESPCN         |   Bicubic    | Original |
| ------------- | ------------- | ------------- |------------- | ------------- |
|Set 14|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/LR_X4/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/UPSCALE_X4/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/BICUBIC_X4/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set14/comic.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/LR_X4/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/UPSCALE_X4/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/BICUBIC_X4/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set5/butterfly.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/LR_X4/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/UPSCALE_X4/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/BICUBIC_X4/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/celeba/000010.png)|

### Upscale x5
|Dataset| Input        | ESPCN         |   Bicubic    | Original |
| ------------- | ------------- | ------------- |------------- | ------------- |
|Set 14|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/LR_X5/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/UPSCALE_X5/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set14/BICUBIC_X5/comic.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set14/comic.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/LR_X5/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/UPSCALE_X5/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/Set5/BICUBIC_X5/butterfly.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/Set5/butterfly.png)|
|Set 5|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/LR_X5/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/UPSCALE_X5/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/results/celeba/BICUBIC_X5/000010.png)|![](https://github.com/ywang530/ESPCN/blob/main/data/test/celeba/000010.png)|

##  Analysis
| Evaluation PSNR | PSNR line fitting |
| ------------- | ------------- |
|![](https://github.com/ywang530/ESPCN/blob/main/analysis/Eval%20PSNR.png)|![](https://github.com/ywang530/ESPCN/blob/main/analysis/fitting.png)|

| Set 5 | Set 14 | CelebA |
| ------------- | ------------- | ------------- |
|![](https://github.com/ywang530/ESPCN/blob/main/analysis/set%2014.png)|![](https://github.com/ywang530/ESPCN/blob/main/analysis/set5.png)| ![](https://github.com/ywang530/ESPCN/blob/main/analysis/celeba.png)
