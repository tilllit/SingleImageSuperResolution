# SingleImageSuperResolution
<br />

This project will investigate on quality disparitys between different approaches to image Super-Resolution by using different metrics. <br />
Super-Resolution is the process of increasing the resolution of an image and therefore the contained information. <br />

The following approaches will be discussed:
- **Interpolation**
- **CNN** (Convolutional neural networks)
- **GAN** (Generative adversarial networks)
<br />
<br />


## Metrics (image quality)

The following metrics ase used to determine the quality of the different SR- (Super Resolution) techniques:
- **PSNR** (Peak Signal Noide Ratio) <br />
  PSNR is a long established image quality metric, most commonly used to compare the compression of different codecs, such as image compression. <br />

  <p align="left">
  <img height="140" width="250" src="figures/psnr.png">
  </p>
  
  - The **HIGHER** the PSNR, the better the image. <br /><br />
  
- **SSIM** (Structural Similarity Index Measure) <br />
  The SSIM compares the luminance, contrast and structure of the original and degraded image. It measures the structural elements of the pixels.
  The SSIM is calculated between 0 and 1, while 1 beeing the best possible value.
  
  <p align="left">
  <img height="140" width="250" src="figures/ssim.png">
  </p>

  - The **HIGHER** the SSIM, the better the image. <br /><br />
  
- **MSE** (Mean Squared Error) <br />
  The MSE as the name suggests is an error (between estimation and target) and is therefor to be minimized. The MSE is widely known accross the machine learning scene. <br /><br />
  
  - The **LOWER** the MSE, the better the image. <br /><br />
  
- **Laplace-Algorithm** for Sharpness / blurriness
  The laplacian algorithm can be used as a metric for blurrinness in an image, with higher values beeing less blurry. <br />
  
  - The **HIGHER** the Sharpness, the better the image.
<br />

[Source: PSNR, SSIM](https://www.testdevlab.com/blog/full-reference-quality-metrics-vmaf-psnr-and-ssim)
 / 
[Source: Laplace](https://medium.com/@sagardhungel/laplacian-and-its-use-in-blur-detection-fbac689f0f88)
<br />
<br />


## Interpolation

The first technique for image Super-Resolution is interpolation. <br />
Standard methods for interpolation are algorithms like: <br />
- **Bi-Linear** interpolation
- **Nearest neighbot** interpolation
- **Bi-Cubic** interpolation
- **Lanczos** interpolation
<br />

The following picture show a visual representaion of the operational principle of the specific algorithms: 
[Source](https://matplotlib.org/1.4.2/examples/images_contours_and_fields/interpolation_methods.html)

<p align="left">
  <img src="figures/Interpolation_visual.png">
</p>
<br />

### Application:

<p align="left">
  <img src="figures/Interpolation.png">
</p>

The metrics imply a close race between bicubic and lanczos algorithm. The descision is made in favour of the **LANCZOS**, because here the strengths are more about visual quality than simmilarity to the original image.

So in all following comparisons the **LANCZOS** interpolation algorithm is representative of the interpolation technique. <br />

A visual comparison between the interpolation algorithms as seen in the picture can be dublicated by the user running the "Interpolation.py" script in the folder [Interpolation](https://github.com/tilllit/SingleImageSuperResolution/tree/main/Interpolation)
<br />
<br />


## Dataset (DIV2K)

For training the DIV2k dataset was used, as it contains high resolution images of landscapes, humans, achitecture and more... <br />
Therefore it should work best on a wide spectrum of applications.

"The DIV2K dataset is one of the most popular datasets used for image super-resolution, which is collected for NTIRE2017 and NTIRE2018 Super-Resolution Challenges. The dataset is composed of 800 images for training, 100 images for validation, and 100 images for testing. Each image has a 2K resolution." - [Medium](https://openmmlab.medium.com/awesome-datasets-for-super-resolution-introduction-and-pre-processing-55f8501f8b18)

### Download:
The dataset can be downloaded from this link: [Kaggle](https://www.kaggle.com/datasets/rain0905/div2k-dataset)
<br />
<br />

## SRCNN

The architectur of the SRCNN (Super Resolution Convolutional Neural Network) follows the description of the following papers:
- [Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
- [Image Super-Resolution Using Deep
Convolutional Networks](https://arxiv.org/abs/1501.00092)

<br />

<p align="center">
  <img height="200" width="800" src="figures/srcnn_architecture.png">
</p>

The training of a SRCNN that produces higher quality output images can take more than 10^8 epochs and is therefore very time intensive as mentioned in the above papers. <br />

The results of a medium-range performing network is shown in the following picture (4k epochs):

<p align="center">
  <img height="350" width="500" src="figures/lanczos_vs_cnn.png ">
</p>

<br />
As the figure points out, the network produces a better PSNR and SSIM score. The MSE is worse in this case, probably because of the higher generative stake. The laplace algorithm rates the network to be more blurry, what is expected to decrease with more epochs of training. See following picture [Source](https://github.com/YeongHyeon/Super-Resolution_CNN):

<p align="center">
  <img height="640" width="960" src="figures/GAN_div2k.gif">
</p>


## SRGAN
<p align="center">
  <img height="640" width="960" src="figures/GAN_div2k.gif">
</p>

###

* end
