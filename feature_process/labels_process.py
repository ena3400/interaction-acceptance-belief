import os
import pympi
import sys
import numpy as np
import cv2
from tools.utils import read_json_file, get_file_name, is_ordered
from tools.video_utils import get_video_fps, get_video_duration

sys.path.append("../../")


def get_elan(path_to_file):
    eaf = pympi.Elan.Eaf(path_to_file)
    return eaf


####################################################


def get_label_segments(eaf):
    """
    :param eaf:
    :return: list of segment [{'start': 6, 'end': 759, 'id': [24, 25], 'iab': [4, 4]}, ...]
    """
    segments = []
    for k_participant, v in eaf.tiers["Participant"][0].items():
        for v_eng in eaf.tiers["Engagement-level"][1].values():
            if k_participant in v_eng:
                segment = {}
                segment["start"], segment["end"] = eaf.timeslots[v[0]], eaf.timeslots[v[1]]
                segment["id"] = [int(i) for i in v[2].split()]
                # print(v_eng[1])
                segment["iab"] = [int(i) for i in v_eng[1].split()]
                segments.append(segment)
    return segments


# Convert segment labels to binary representation
def convert_to_binary(labels, total_duration):
    binary_labels = [0] * total_duration
    for segment in labels:
        start, end = segment
        binary_labels[start:end + 1] = [1] * (end - start + 1)
    return binary_labels


def convert_to_label_per_frame(label_segments, tracking_per_frame, fps):
    labels = []
    t = 0
    for tracking in tracking_per_frame["data"]:
        if len(tracking) != 0:
            # compare with label
            empty = 1
            for segment in label_segments:
                if segment["start"] <= t < segment["end"]:
                    labels.append(segment["iab"])
                    empty = 0
                    break
            # add 3 for unlabeled tracking
            if empty == 1:
                labels.append([3])
        t += (1 / fps) * 1000
    return labels


def compute_cohen_kappa(eaf1_path, eaf2_path, json_path, video_path):
    tracking_per_frame = read_json_file(json_path)
    eaf1 = get_elan(eaf1_path)
    eaf2 = get_elan(eaf2_path)
    segments_1 = get_label_segments(eaf1)
    segments_2 = get_label_segments(eaf2)
    fps = get_video_fps(video_path)
    print(fps)
    labels1 = convert_to_label_per_frame(segments_1, tracking_per_frame, fps)
    labels2 = convert_to_label_per_frame(segments_2, tracking_per_frame, fps)
    assert len(labels1) == len(labels2)
    same = 0
    for i in range(len(labels1)):
        if labels1[i] == labels2[i]:
            same += 1
    return same / len(labels1)


def compute_kappa_on_all_data():
    video_names = os.listdir("../datasets/home_data/video")
    kappa_mean = 0
    for video_name in video_names:
        print(video_name)
        video = video_name.split(".")[0]
        path_to_file = f'datasets/home_data/labels/tim/{video}_processed.eaf'
        path_to_file2 = f'datasets/home_data/labels/mathys/{video}_processed.eaf'
        json_path = "datasets/home_data/tracking/" + video + "_processed.json"
        video_path = "datasets/home_data/video/" + video + ".MOV"
        kappa_mean += compute_cohen_kappa(path_to_file, path_to_file2, json_path, video_path)
        print(compute_cohen_kappa(path_to_file, path_to_file2, json_path, video_path))
    return kappa_mean / len(video_names)


if __name__ == "__main__":
    # kappa = compute_kappa_on_all_data()
    # print(kappa)
    print("end")
