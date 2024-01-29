import torch
from PIL import Image
import uuid
import os
import numpy as np
from tqdm import tqdm
import time

from lavis.models import load_model_and_preprocess
from lavis.datasets.data_utils import load_video, load_clip

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="spotlight", is_eval=True, device=device)
model.eval()
# model.float()

from PIL import Image
import cv2

def save_npz(features_buffer, media_ids, output_path):

    filename = os.path.join(output_path, str(uuid.uuid4()) + '.npz')

    # print(len(features_buffer))
    # print(features_buffer[0])

    features = np.concatenate(features_buffer, axis=0)
    # media_ids = np.concatenate(media_ids_buffer, axis=0)
    with open(filename, 'wb') as f:
        np.savez_compressed(f, features=features, media_ids=media_ids)

visual_outputs = []
media_ids = []

output_path = '/nfs/swang7/500k_db/blip2_features_20240119072/'

input_list = '/nfs/swang7/500k_db/inference_8/list_7.csv'
video_dir = '/nfs/swang7/500k_db/videos'
with open(input_list, 'r') as fp:
    lines = fp.readlines()
video_ids = [line.strip() for line in lines]

start = time.time()
for video_id in tqdm(video_ids):
    video_path = f'{video_dir}/{video_id}.mp4'
    # print(video_path)
    # frames = extract_frames(video_path, num_frames=10)
    # frames = [vis_processors["eval"](frame) for frame in frames]
    # frames = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    try:
        frames = vis_processors["eval"](video_path).unsqueeze(0).to(device)
    except:
        print(video_id)
        continue
    # print(frames.shape)

    sample =  {"video": frames}

    features_image = model.extract_features(sample, mode="video") # (1, 320, 768)
    features_image = torch.mean(features_image.image_embeds, dim=1) # (1, 768)
    image_features = features_image / features_image.norm(dim=-1, keepdim=True) # (1, 768)
    # print(image_features.shape)

    visual_outputs.append(image_features.detach().cpu().numpy())
    media_ids.append(video_id)
    if len(media_ids) >= 2000:
        save_npz(visual_outputs, media_ids, output_path)
        visual_outputs = []
        media_ids = []

        end = time.time()
        print(f'time elapsed for a batches: {end - start}')
        start = time.time()
        # break
if len(media_ids) > 0:
    save_npz(visual_outputs, media_ids, output_path)