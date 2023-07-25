import os

import numpy as np
from pathlib import Path
import argparse
import cv2

import sys

sys.path.append('../')
from tools.utils import read_json_file, check_and_create_folder, save_json
import torch
from .emonet.models import EmoNet


def load_emonet_model(model_path):
    device = 'cuda:0'
    # Loading the model
    state_dict_path = model_path
    print(f'Loading the model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=8).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


def get_feat_vector(image_path, model):
    """
    For 8 expression :
    0 - Neutral
    1 - Happy
    2 - Sad
    3 - Surprise
    4 - Fear
    5 - Disgust
    6 - Anger
    7 - Contempt
    return numpy vector [Neutral,Happy,Sad,Surprise,Fear,Disgust,Anger,Contempt,valence,arousal]
    """
    device = 'cuda:0'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256)).transpose([2, 0, 1])
    image = np.asarray(image, dtype=np.float32)
    image = torch.tensor(image).unsqueeze(0)
    with torch.no_grad():
        out = model(image.to(device))
    # Concatenate the tensors
    concatenated_tensor = torch.cat((out["expression"][0], out["valence"], out["arousal"]), dim=0)
    concatenated_tensor = concatenated_tensor.to('cpu').detach().numpy()
    return concatenated_tensor.tolist()


def process_video_emotion_feat(dataset_path, video_name, model):
    json_userimgid = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json")
    check_and_create_folder(f"{dataset_path}/emotion/{video_name}/")
    user_name_list = json_userimgid.keys()
    for user in user_name_list:
        json_user_to_save = {}
        image_list = os.listdir(f"{dataset_path}/user_img/{video_name}/{user}")
        image_id = json_userimgid[user]
        imgfile2imgid = {}
        # track the original id of the image
        for imgfile, imgid in enumerate(image_id):
            imgfile2imgid[imgfile] = imgid
        # process every image
        for img_file in image_list:
            image_path = f"{dataset_path}/user_img/{video_name}/{user}/{img_file}"
            emotion_feat = get_feat_vector(image_path, model)
            img_file = int(img_file.split(".")[0])
            json_user_to_save[str(imgfile2imgid[img_file])] = emotion_feat
        save_json(json_user_to_save, f"{dataset_path}/emotion/{video_name}/{user}_emotion_feat.json")


if __name__ == "__main__":
    img_path = "../datasets/home_data/user_img/IMG_0001/user1/0.png"
    model = load_emonet_model()
    a = get_feat_vector(img_path, model)
    print(a)
