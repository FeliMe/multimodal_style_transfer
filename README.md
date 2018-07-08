# Deep Neural Style Transfer
This is a PyTorch implementation of the paper ['Multimodal Transfer: A Hierarchical Deep Convolutional Neural Network for Fast Artistic Style Transfer'](https://arxiv.org/abs/1612.01895) by Wang et al.

## Usage
```
$ git clone https://gitlab.lrz.de/ge56qib/Seminar_CTDL/tree/master
```

## Transfer styles
if you just want to use the network with the pretrained models, open 'transform_image.ipynb' (or 'transform_video.ipynb'), select the model and an image from the /images folder (or use your own) and run the notebook.

## Train
If you want to train your own model on a styles image, you first need to download the [MS COCO Dataset](http://cocodataset.org/#download), store it in a folder named "/coco/" in the same directory where you cloned this project to. Then use 'train_multimodal.ipynb'. You might need to adapt the STYLE_WEIGHTS depending on you style image.

## Examples
|Patch|<img src='styles/patch' width='256px'>|<img_scr="generated_images/multimodal_patch_256.jpg">|<img_scr="generated_images/multimodal_patch_512.jpg">|<img_scr="generated_images/multimodal_patch_1024.jpg">|
|Scream|<img src='styles/scream' width='256px'>|<img_scr="generated_images/multimodal_scream_256.jpg">|<img_scr="generated_images/multimodal_scream_512.jpg">|<img_scr="generated_images/multimodal_scream_1024.jpg">|
|Still life|<img src='styles/still_life' width='256px'>|<img_scr="generated_images/multimodal_still_life_256.jpg">|<img_scr="generated_images/multimodal_still_life_512.jpg">|<img_scr="generated_images/multimodal_still_life_1024.jpg">|
|Mixed still life & starry night|<img src='styles/starry_night' width='256px'>|<img_scr="generated_images/multimodal_mixed_still_life_starry_night_256.jpg">|<img_scr="generated_images/multimodal_mixed_still_life_starry_night_512.jpg">|<img_scr="generated_images/multimodal_mixed_still_life_starry_night_1024.jpg">|

## Implementation Details
There are some deviations from the original paper in this implementation. I chose a different layer for the content representation as it generated better results. I also added a regularization loss as in [Johnson et al.](https://arxiv.org/abs/1603.08155). 

I removed the final bilinear upsampling to size 1024 as it was only inserted during test time anyway and thus resulted in an image which didn't have higher effective resolution but was just an upsampled version of the previous image.

## Acknowledgements
- Some code is based on the fast neural style transfer implementation by [CeShine Lee](https://github.com/ceshine/fast-neural-style)
- Thanks to the [TUM Computer Vision Group](https://vision.in.tum.de/) for granting me access to their hardware.
