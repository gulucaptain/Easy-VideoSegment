# Easy Video & Image Semantic Segmentation

<p align="center">
<img src="https://github.com/user-attachments/assets/9cda61ff-f4b6-4ed2-abe6-45ac0a1d8302" width="880">
</p>

## Introduction of this project

* We provide an easy function to use Segment and Track Anything for video and image semantic segmentation.
* We gather the needed checkpoingts in a third-party huggingface link and only one ckpt_download.sh script can easily prepare the all checkpoints.
* We provide the directly test Python file for inference, you only need to define the path of a single image or a image folders, then you can obtain the semantic segmentation results.

## About Segment and Track Anything

**Segment and Track Anything** (https://github.com/z-x-yang/Segment-and-Track-Anything) is an open-source project that focuses on the segmentation and tracking of any objects in videos, utilizing both automatic and interactive methods. The primary algorithms utilized include the [**SAM** (Segment Anything Models)](https://github.com/facebookresearch/segment-anything) for automatic/interactive key-frame segmentation and the [**DeAOT** (Decoupling features in Associating Objects with Transformers)](https://github.com/yoxu515/aot-benchmark) (NeurIPS2022) for efficient multi-object tracking and propagation. The SAM-Track pipeline enables dynamic and automatic detection and segmentation of new objects by SAM, while DeAOT is responsible for tracking all identified objects.

## Start

### 1. Installing envs
Run following script to create conda env and install the dependencies.
```
bash install.sh
```

Add the absolute path of the groundingdino folder in line 7 of tool/detector.py.
```
sys.path.append('/home/user/xxx/Easy-VideoSegment/src/groundingdino')
```

### 2. Model Preparation
We need SAM model (SAM-VIT-B), DeAOT/AOT model (R50-DeAOT-L), Grounding-Dino model, and AST model (audioset_0.4593).

You can download the default weights using the command line as shown below (remember to install huggingface_hub).
```
bash ckpt_download.sh
```

### 3. Running Demo
You only need to define three things in inference.sh:

* file_pth: the path of single image/video path or the folder  of images/videos.
* segment_label: the pixel parts in image or video you want to segment.
* mask_save_pth: the save path.

Then, run:
```
bash inference.sh
```

The results will be saved in <mask_save_pth>.

### Citations
```
@article{cheng2023segment,
  title={Segment and Track Anything},
  author={Cheng, Yangming and Li, Liulei and Xu, Yuanyou and Li, Xiaodi and Yang, Zongxin and Wang, Wenguan and Yang, Yi},
  journal={arXiv preprint arXiv:2305.06558},
  year={2023}
}
```
