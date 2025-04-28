import os
from PIL import Image
from pathlib import Path
from operator import itemgetter
import numpy as np
import random
import cv2
import time
import argparse
import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor

from model_args import aot_args,sam_args,segtracker_args
from SegTracker import SegTracker
from aot_tracker import _palette

import gc
import imgviz
colormap = imgviz.label_colormap(80)

import warnings
warnings.filterwarnings("ignore")


def return_video_frames(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)

    return save_mask

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    file_name = file_name.split("_")[0] + ".png"
    save_mask.save(os.path.join(output_dir,file_name))

def sparse_sample(total_frames, sample_frames, sample_rate, random_sample=False):
    if sample_frames <= 0:  # sample over the total sequence of frames
        ids = np.arange(0, total_frames, sample_rate, dtype=int).tolist()
    elif sample_rate * (sample_frames - 1) + 1 <= total_frames:
        offset = random.randrange(total_frames - (sample_rate * (sample_frames - 1))) \
            if random_sample else 0
        ids = list(range(offset, total_frames + offset, sample_rate))[:sample_frames]
    else:
        ids = np.linspace(0, total_frames, sample_frames, endpoint=False, dtype=int).tolist()
    return ids

def video_to_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        frames.append(frame_tensor)
    cap.release()
    video_tensor = torch.cat(frames, dim=0)
    return video_tensor, fps

def images_to_video(image_list, output_video_path, fps):
    width, height = image_list[0].size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img in image_list:
        img_bgr = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    video_writer.release()

def load_images(file_path, n_sample_frames, sample_rate=4, random_sample=False, transform=None):
    sample_args = dict(sample_frames=n_sample_frames, sample_rate=sample_rate, random_sample=random_sample)
    video = []
    if Path(file_path).is_dir():
        image_numbers = len(os.listdir(file_path))
        img_files = []
        for i in range(0, image_numbers):
            img_files.append(os.path.join(file_path, "frame_"+str(i)+".jpg"))
        
        if len(img_files) < 1:
            print("No data in video directory")
        sample_ids = sparse_sample(len(img_files), **sample_args)
        for img_file in itemgetter(*sample_ids)(img_files):
            img = pil_loader(Path(img_file).as_posix())
            img = pil_to_tensor(img)
            video.append(img)
    video = torch.stack(video)  # (f, c, h, w)
    if transform is not None:
        video = transform(video)
    return video

segtracker_args = {
    'sam_gap': 5, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 1, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
}

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--file_pth', type=str, default='', help='File pth including imgs and videos.')
    parser.add_argument('--segment_label', type=str, default='', help='what do you want to segment?')
    parser.add_argument('--mask_save_pth', type=str, default='', help='save path')

    args = parser.parse_args()

    if os.path.isdir(args.file_pth):
        file_names = os.listdir(args.file_pth)
        file_names.sort()
        files = [os.path.join(args.file_pth, file_name) for file_name in file_names]
    elif os.path.isfile(args.file_pth):
        files = [args.file_pth]
    else:
        assert os.path.isdir(args.file_pth) or os.path.isfile(args.file_pth)
    
    for file in files:
        file_name = os.path.basename(file)
        begin_time = time.time()
        output_mask_path = os.path.join(args.mask_save_pth, file_name.split(".")[0])
        os.makedirs(output_mask_path, exist_ok=True)
        
        video, fps = video_to_tensor(file)

        # generate mask
        pred_list = []
        masked_pred_list = []

        torch.cuda.empty_cache()
        gc.collect()
        sam_gap = segtracker_args['sam_gap']
        segtracker = SegTracker(segtracker_args,sam_args,aot_args)
        segtracker.restart_tracker()
        frame_idx = 0
        box_threshold = 0.25
        text_threshold = 0.25
        
        grounding_captions = [args.segment_label]
        with torch.cuda.amp.autocast():
            for grounding_caption in grounding_captions:
                frame_idx = 0
                for frame in video:
                    frame = frame.permute(1, 2, 0).numpy()
                    if frame_idx == 0:
                        pred_mask, annotated_frame = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
                        torch.cuda.empty_cache()
                        gc.collect()
                        if np.all(pred_mask==0):
                            grounding_captions.remove(grounding_caption)
                            break
                        segtracker.add_reference(frame, pred_mask)
                    else:
                        pred_mask = segtracker.track(frame, update_memory=True)

                    torch.cuda.empty_cache()
                    gc.collect()
                    save_prediction(pred_mask,output_mask_path,str(frame_idx)+ '_' + str(grounding_caption)+ '.png')

                    frame_idx += 1
        end_time = time.time()
        print(f"Time cost: {end_time-begin_time}")
    