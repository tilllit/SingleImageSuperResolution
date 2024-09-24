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
  PSNR is a long established image quality metric, most commonly used to compare the compression of different codecs, such as image compression.
  <p align="center">
  <img src="figures/Interpolation_visual.png">
  </p>
- **SSIM** (Structural Similarity Index Measure)
- **MSE** (Mean Squared Error)
- **Laplace-Algorithm** for blurriness / clarity
<br />

A visual comparison between the interpolation algorithms as seen in the picture can be dublicated by the use with the  using the "Interpolation.py" script in the folder [Interpolation](https://github.com/tilllit/SingleImageSuperResolution/tree/main/Interpolation)
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
[ -Source](https://matplotlib.org/1.4.2/examples/images_contours_and_fields/interpolation_methods.html)

<p align="left">
  <img src="figures/Interpolation_visual.png">
</p>
<br />

By applying the discussed metrics onto an test image upscaled by the mentioned interpolation methods there is the following result:

<p align="left">
  <img src="figures/Interpolation.png">
</p>

The metrics imply a close race between bicubic and lanczos algorithm. <br />
The descision is made in favour of the **LANCZOS**, because here the strengths are more about visual quality than simmilarity to the original image.

So in all following comparisons the **LANCZOS** interpolation algorithm is representative of the interpolation technique.
<br />
<br />


## Dataset (DIV2K)

For training the DIV2k dataset was used, as it contains high resolution images of landscapes, humans, achitecture and more... <br />
Therefore it should work best on a wide spectrum of applications.

"The DIV2K dataset is one of the most popular datasets used for image super-resolution, which is collected for NTIRE2017 and NTIRE2018 Super-Resolution Challenges. The dataset is composed of 800 images for training, 100 images for validation, and 100 images for testing. Each image has a 2K resolution." - [Medium](https://openmmlab.medium.com/awesome-datasets-for-super-resolution-introduction-and-pre-processing-55f8501f8b18)


The dataset can be downloaded from this link: <br />
[Kaggle](https://www.kaggle.com/datasets/rain0905/div2k-dataset)
<br />
<br />


## GAN
<p align="center">
  <img height="640" width="960" src="figures/GAN_div2k.gif">
</p>

###

* end
