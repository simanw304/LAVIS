import torch
from PIL import Image
import uuid
import os
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
import cv2
import json

from lavis.models import load_model_and_preprocess
from lavis.datasets.data_utils import load_video, load_clip

gt_file = '/nfs/swang7/blip2_eval/spotlight-first-9k-testdata-sep.json'

visual_outputs = []
media_ids = []

video_dir = '/nfs/swang7/17k_videos'
with open(gt_file, 'r') as fp:
    data = json.load(fp)


video_ids = [item['video'] for item in data]
# print(len(video_ids))
# print(video_ids[0])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="spotlight", is_eval=True, device=device)
model.eval()
# model.float()

def extract_middle_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the middle frame index
    middle_frame_index = total_frames // 2

    # Set the video capture to the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    # Read the middle frame
    ret, frame = cap.read()

    # Release the video capture object
    cap.release()

    return Image.fromarray(frame)

output = []
for video_id in tqdm(video_ids):
    video_path = f'{video_dir}/{video_id}'

    frames = vis_processors["eval"](video_path).unsqueeze(0).to(device)
    frames = torch.zeros(frames.shape, dtype=frames.dtype).to(device)
    sample =  {"video": frames}

    # middle_frm = extract_middle_frame(video_path)
    # frame = vis_processors["eval"](middle_frm).unsqueeze(0).to(device)
    # sample =  {"image": frame}
    captions = model.generate(sample, num_beams=5)
    output.append({'video': video_id, 'caption': captions})
    print(output)

file_path = "/nfs/swang7/blip2_eval/pred_test.ndjson"

# Open the file for writing
with open(file_path, 'w') as file:
    # Iterate over each dictionary in the data list
    for item in output:
        # Serialize the dictionary to a JSON string and write it to the file with a newline
        json.dump(item, file)
        file.write('\n')
