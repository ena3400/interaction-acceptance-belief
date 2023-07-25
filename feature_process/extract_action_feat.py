import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append('../')
from tools.utils import read_json_file, check_and_create_folder, save_json
from .i3d.pytorch_i3d import InceptionI3d
from utils.datasets import letterbox


def pad_to_square(image):
    height, width, _ = image.shape
    max_dim = max(height, width)

    # Calculate padding values
    pad_top = int((max_dim - height) / 2)
    pad_bottom = max_dim - height - pad_top
    pad_left = int((max_dim - width) / 2)
    pad_right = max_dim - width - pad_left

    # Pad the image with border
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return padded_image


def load_model(model_path, dataset):
    # Load the I3D model
    i3d = InceptionI3d(400, in_channels=3)
    if dataset == "charades":
        i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.cuda()
    i3d.train(False)
    return i3d


def get_sequences(folder_path, sequence_length, overlap, image_extensions=('.jpg', '.png')):
    # Get sorted list of image files
    # image_files = sorted([
    #     file for file in os.listdir(folder_path)
    #     if os.path.splitext(file)[1].lower() in image_extensions
    # ])
    # image_files = sorted([int(id_.split(".")[0]) for id_ in image_files])
    user = folder_path.split("/")[-1]
    json_path = folder_path.split(user)[0]
    image_id = read_json_file(json_path + "user_id2img.json")[user]
    # print(image_files)
    # Extract features for each sequence
    sequences = []
    current_sequence = []
    previous_frame = None
    sequence_fill = 0
    for frame_number in image_id:

        if previous_frame is not None:
            remaining_sequence_size = sequence_length - len(current_sequence)
            # si il y a un trou entre les 2 et que la frame d'arrivée est toujours dans la sequence le combler
            if remaining_sequence_size > frame_number - previous_frame > 1:
                # Handle missing frames by filling with the previous frame
                missing_frames = frame_number - previous_frame - 1
                for _ in range(missing_frames):
                    sequence_fill += 1
                    current_sequence.append(previous_frame)
            # si il y a un trou entre les 2 qui dépasse la taille de la séquence actuelle
            if frame_number - previous_frame > remaining_sequence_size:
                # remplir le rest
                for i in range(len(current_sequence), sequence_length):
                    current_sequence.append(current_sequence[-1])
                    sequence_fill += 1
                    # si plus de moins de 30% de la sequence n'a pas été fill sauvegarder
                if sequence_fill < int(sequence_length * 0.20) and len(current_sequence) == sequence_length:
                    sequences.append(current_sequence)
                sequence_fill = 0
                current_sequence = []

        current_sequence.append(frame_number)

        if len(current_sequence) == sequence_length and sequence_fill < int(sequence_length * 0.20):
            sequences.append(current_sequence)
            sequence_fill = 0
            if overlap <= 0:
                current_sequence = []
            else:
                current_sequence = current_sequence[-overlap:]

        previous_frame = frame_number
    return sequences


def get_features(sequences, folder_path, model, prediction=False):
    user_name = folder_path.split("/")[-1]
    json_path = folder_path.split(user_name)[0]
    image_id = read_json_file(json_path + "user_id2img.json")[user_name]
    imgid2imgfile = {}
    for imgfile, imgid in enumerate(image_id):
        imgid2imgfile[imgid] = f"{imgfile}.png"
    sequence_features = []
    sequence_prediction = []
    # Process each sequence and extract features
    for sequence in sequences:
        sequence_frames = []
        for img_id in sequence:
            image_path = os.path.join(folder_path, imgid2imgfile[img_id])
            frame = cv2.imread(image_path)
            frame = pad_to_square(frame)
            # Resize the image
            frame = cv2.resize(frame, (224, 224)).transpose([2, 0, 1])
            # Add the preprocessed frame to the sequence_frames list
            sequence_frames.append(frame)
        sequence_frames = np.asarray(sequence_frames, dtype=np.float32)
        sequence_frames = torch.tensor(sequence_frames).unsqueeze(0).permute(0, 2, 1, 3, 4).cuda()

        # Check the currently allocated GPU memory
        # allocated_memory = torch.cuda.memory_allocated()
        # print(f"Currently Allocated GPU Memory: {allocated_memory / 1024 ** 2:.2f} MB")

        features = model.extract_features(sequence_frames).squeeze(-2).squeeze(-1).to('cpu').detach().numpy()
        if prediction:
            predictions = model.forward(sequence_frames).to('cpu').detach().numpy()
            sequence_prediction.append(predictions)
        sequence_features.append(features)
        del sequence_frames
    return sequence_features, sequence_prediction


def save_features(video_name, model, dataset_path):
    # user_name_list = os.listdir(f"{dataset_path}/user_img/{video_name}")
    user_name_list = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json").keys()
    check_and_create_folder(f"{dataset_path}/action_features/{video_name}")
    for user_name in user_name_list:
        folder_path = f"{dataset_path}/user_img/{video_name}/{user_name}"
        sequences = get_sequences(folder_path, 64, 0)  # 30 fps
        sequence_features, _ = get_features(sequences, folder_path, model)
        check_and_create_folder(f"{dataset_path}/action_features/{video_name}/{user_name}")
        sequenceid2sequence = {}
        for id_seq, sequence in enumerate(sequences):
            sequenceid2sequence[id_seq] = sequence
            np.savez(f"{dataset_path}/action_features/{video_name}/{user_name}/{id_seq}.npz", sequence_features[id_seq])
        save_json(sequenceid2sequence, f"{dataset_path}/action_features/{video_name}/{user_name}_sequences.json")
        # To load the saved data
        # loaded_data = np.load('data.npz', allow_pickle=True)


def load_predictions(video_name, model, dataset_path):
    # user_name_list = os.listdir(f"{dataset_path}/user_img/{video_name}")
    user_name_list = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json").keys()
    check_and_create_folder(f"{dataset_path}/action_features/{video_name}")
    for user_name in user_name_list:
        folder_path = f"{dataset_path}/user_img/{video_name}/{user_name}"
        sequences = get_sequences(folder_path, 64, 0)  # 30 fps
        sequence_features, predictions = get_features(sequences, folder_path, model)
        check_and_create_folder(f"{dataset_path}/action_features/{video_name}/{user_name}")
        sequenceid2sequence = {}
        for id_seq, sequence in enumerate(sequences):
            sequenceid2sequence[id_seq] = sequence
            np.savez(f"{dataset_path}/action_features/{video_name}/{user_name}/{id_seq}.npz", sequence_features[id_seq])
        save_json(sequenceid2sequence, f"{dataset_path}/action_features/{video_name}/{user_name}_sequences.json")


def watch_features(video_name, model, dataset_path, dataset):
    # user_name_list = os.listdir(f"{dataset_path}/user_img/{video_name}")
    user_name_list = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json").keys()
    check_and_create_folder(f"../visualization/action/{dataset}/")
    check_and_create_folder(f"../visualization/action/{dataset}/{video_name}")
    # get Charade classes
    if dataset == "charades":
        c_numbers = []
        sentences = []
        with open("../datasets/charades/Charades/Charades_v1_classes.txt", "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    c_number, sentence = line.split(" ", 1)
                    c_numbers.append(c_number)
                    sentences.append(sentence)
    if dataset == "imagenet":
        df = pd.read_csv("../datasets/kinetics/kinetics_400_labels.csv", index_col="id")
        sentences = df["name"].tolist()
    for user_name in user_name_list:
        folder_path = f"{dataset_path}/user_img/{video_name}/{user_name}"
        sequences = get_sequences(folder_path, 64, 0)  # 30 fps
        sequence_features, predictions = get_features(sequences, folder_path, model, prediction=True)
        sequenceid2sequence = {}
        for id_seq, sequence in enumerate(sequences):
            sequenceid2sequence[id_seq] = sequence
        # save img with label
        user_id_list = read_json_file(f"{dataset_path}/person_id/{video_name}.json")
        bbox_per_frames = read_json_file(dataset_path + "/tracking/" + video_name + "_processed.json")["data"]
        for id_seq, sequence in enumerate(sequences):
            img_name_list = []
            for img_id in sequence:
                for img_name in os.listdir(f"{dataset_path}/images/{video_name}"):
                    if img_name.split('_')[0] == str(img_id):
                        img_name_list.append(img_name)
            for img_name in img_name_list:
                if os.path.exists(f"../visualization/action/{dataset}/{video_name}/{img_name}"):
                    frame = cv2.imread(f"../visualization/action/{dataset}/{video_name}/{img_name}")
                else:
                    frame = cv2.imread(f"{dataset_path}/images/{video_name}/{img_name}")
                    frame = letterbox(frame, (640), stride=64, auto=True)[0]
                for tracking in bbox_per_frames[int(img_name.split("_")[0])]:
                    if tracking["id"] in user_id_list[user_name]:
                        bbox = tracking["bbox"]
                prediction = np.squeeze(np.mean(predictions[id_seq], axis=-1))
                prediction = np.argmax(prediction)
                sentence_pred = sentences[prediction]
                cv2.putText(frame, sentence_pred, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                cv2.imwrite(f"../visualization/action/{dataset}/{video_name}/{img_name}", frame)


if __name__ == "__main__":
    model_dataset = "imagenet"
    model_path = f"i3d/models/rgb_{model_dataset}.pt"
    model = load_model(model_path,model_dataset)
    for video_name in os.listdir("../datasets/home_data/images"):
        dataset_path = "../datasets/home_data/"
        watch_features(video_name, model, dataset_path, model_dataset)
    # # Check the currently allocated GPU memory
    # allocated_memory = torch.cuda.memory_allocated()
    # print(f"Currently Allocated GPU Memory: {allocated_memory / 1024 ** 2:.2f} MB")
    # video_list = os.listdir("../datasets/home_data/images")
    # for video in video_list:
    #     save_features(video, model)
    # loaded_data = np.load(f"../datasets/home_data/action_features/IMG_0001/user1_action_feats.npz",
    #                      allow_pickle=True)
    # Access the dictionaries from the loaded data
    # dict1 = loaded_data['arr_0']
    # dict2 = loaded_data['arr_1']
    print("lol")
