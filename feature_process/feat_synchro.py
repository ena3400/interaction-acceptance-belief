import os
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from tools.utils import read_json_file, check_and_create_folder, save_json
from tools.video_utils import get_video_duration
from tools.bbox_utils import normalize_keypoints, project_keypoints, is_point_in_bbox
from .labels_process import get_label_segments, get_elan


def save_load_img2time(dataset_path, video_name):
    if os.path.exists(f"{dataset_path}/user_img/{video_name}/imgid2time.json"):
        imgid2time = read_json_file(f"{dataset_path}/user_img/{video_name}/imgid2time.json")
    else:
        imgid2time = {}
        img_time_name_list = os.listdir(f"{dataset_path}/images/{video_name}")
        for img_time_name in img_time_name_list:
            imgid2time[img_time_name.split("_")[0]] = img_time_name.split(".jpg")[0].split("_")[-1]
        save_json(imgid2time, f"{dataset_path}/user_img/{video_name}/imgid2time.json")
    return imgid2time


def get_user_headpose(dataset_path, user_name, video_name):
    """
    :return: time list and corresponding features # check format kp
    """
    bbox_per_frames = read_json_file(dataset_path + "/tracking/" + video_name + "_processed.json")["data"]
    user_id_list = read_json_file(f"{dataset_path}/person_id/{video_name}.json")[user_name]
    image_list = os.listdir(f"{dataset_path}/images/{video_name}")
    sorted_list = sorted(image_list, key=lambda x: int(x.split('_')[0]))
    time_list = []
    feat_list = []
    for i, img_name in enumerate(sorted_list):
        for bbox in bbox_per_frames[i]:
            if bbox["id"] in user_id_list:
                new_features = []
                for i in range(0, len(bbox["kp"]), 3):
                    x, y, confidence = bbox["kp"][i:i + 3]
                    if confidence >= 0.8:
                        new_features.extend([x, y])
                    else:
                        new_features.extend([0, 0])
                new_features = [int(feat) for feat in new_features]
                new_features = new_features[:11 * 2]  # take only till hand
                if sum(new_features) != 0:
                    new_features = normalize_keypoints(new_features)
                feat_list.append(new_features)
                time_list.append(img_name.split(".jpg")[0].split("_")[-1])
    return time_list, feat_list


def get_user_openface_video_with_time(dataset_path, user_name, video_name):
    """
    extract df openface for processed openface video on image with a multiperson tracker
    """
    user_id_list = read_json_file(f"{dataset_path}/person_id/{video_name}.json")  # [user_name]
    bbox_per_frames = read_json_file(dataset_path + "/tracking/" + video_name + "_processed.json")["data"]
    image_list = os.listdir(f"{dataset_path}/images/{video_name}")
    sorted_list = sorted(image_list, key=lambda x: int(x.split('_')[0]))
    time_list = [img_name.split(".jpg")[0].split("_")[-1] for img_name in sorted_list]
    assert len(bbox_per_frames) == len(time_list)
    # load openface
    df_openface = pd.read_csv(f"{dataset_path}/openface/{video_name}.csv")
    # video en 1920 × 1080 mov 640 × 384 tracking
    # Assign openface id to user_id_list
    filtered_df = df_openface[df_openface[' confidence'] > 0.90]  # .iloc[0]
    openfaceid2userid = {}
    for id in df_openface[' face_id'].unique():
        df_id = filtered_df[filtered_df[' face_id'] == id]
        if len(df_id) > 0:
            row = df_id.iloc[0]
            face_time = row[' timestamp']
            face_kp = [row[" p_tx"], row[" p_ty"]]
            closest_time = min(time_list, key=lambda x: abs(float(x) - face_time))
            closest_time_index = time_list.index(closest_time)
            scaled_face_kp = project_keypoints(face_kp, 1920, 1080, 640, 384)
            # check if face kp is in a bbox
            for tracking in bbox_per_frames[closest_time_index]:
                # in_bbox = is_point_in_bbox(scaled_face_kp, tracking["bbox"])
                if is_point_in_bbox(scaled_face_kp, tracking["bbox"]):
                    bbox_id = tracking["id"]
                    for user, bbox_ids in user_id_list.items():
                        if bbox_id in bbox_ids:
                            openfaceid2userid[id] = user
                            break
                    break
    df_openface[" face_id"] = df_openface[' face_id'].map(openfaceid2userid)
    df_user_openface = df_openface[df_openface[" face_id"] == user_name]
    df_user_openface = df_user_openface[df_user_openface[' confidence'] > 0.60]
    df_user_openface = df_user_openface.rename(columns={' timestamp': 'time'})
    return df_user_openface


def get_user_openface_with_time(dataset_path, user_name, video_name):
    """
    :param dataset_path:
    :param user_name:
    :param video_name:
    :return: dataframe with "image_id" and its corresponding "image_time" in columns
    """
    df_openface = pd.read_csv(f"{dataset_path}/openface/{video_name}_{user_name}.csv")
    json_imgfile2id = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json")
    if len(df_openface) != len(json_imgfile2id[user_name]):
        print(f"Error on {video_name} {user_name}")
    # load or create image name from user_img to image id from images
    imgid2time = save_load_img2time(dataset_path, video_name)
    # check if there is image time in dataframe else create it
    if "image_id" not in df_openface.columns or "time" not in df_openface.columns:
        img_time_list = []
        img_openface_id_list = []
        img_user_list = os.listdir(f"{dataset_path}/user_img/{video_name}/{user_name}")
        img_user_list = sorted(img_user_list, key=lambda x: int(x.split('.')[0]))
        for img_userid in img_user_list:
            img_userid = int(img_userid.split(".")[0])
            img_time_list.append(imgid2time[str(img_userid)])
            img_openface_id_list.append(json_imgfile2id[user_name][int(img_userid)])
        # for imgid in json_imgfile2id[user_name]:
        #     img_time_list.append(imgid2time[str(imgid)])
        df_openface["image_id"] = img_openface_id_list
        df_openface["time"] = img_time_list
        # Save DataFrame to CSV
        df_openface.to_csv(f"{dataset_path}/openface/{video_name}_{user_name}.csv", index=False)
    return df_openface


def get_action_reco_with_time(dataset_path, user_name, video_name):
    """
    :return: a list of ordered time of features and a dict time to features
    """
    action_sequences_user_feats = read_json_file(
        f"{dataset_path}/action_features/{video_name}/{user_name}_sequences.json")
    imgid2time = save_load_img2time(dataset_path, video_name)
    seqid2time_feat = {}
    for sequence_id, sequence_img_id in action_sequences_user_feats.items():
        seqid2time_feat[sequence_id] = {}
        seqid2time_feat[sequence_id]["start"] = imgid2time[str(sequence_img_id[0])]
        seqid2time_feat[sequence_id]["end"] = imgid2time[str(sequence_img_id[-1])]
        feats = np.load(f"{dataset_path}/action_features/{video_name}/{user_name}/{sequence_id}.npz")
        feats = feats["arr_0"]
        seqid2time_feat[sequence_id]["feat"] = feats
        time_sequence = []
        for i in range(1, feats.shape[-1] + 1):
            total_time = float(seqid2time_feat[sequence_id]["end"]) - float(seqid2time_feat[sequence_id]["start"])
            timestep = total_time / (feats.shape[-1])
            time_feat = float(seqid2time_feat[sequence_id]["start"]) - timestep / 2 + (i * timestep)
            time_sequence.append(time_feat)
        seqid2time_feat[sequence_id]["time_feats"] = time_sequence
    time2action_feat = {}
    time_list = []
    for seqid in seqid2time_feat.keys():
        for i, time in enumerate(seqid2time_feat[seqid]["time_feats"]):
            time2action_feat[time] = seqid2time_feat[seqid]["feat"][:, :, i]
            time_list.append(float(time))
    time_list = sorted(time_list)
    return time_list, time2action_feat


def get_action_reco_with_time_averaged(dataset_path, user_name, video_name):
    """
    :return: a list of ordered time of features and corresponding features
    """
    action_sequences_user_feats = read_json_file(
        f"{dataset_path}/action_features/{video_name}/{user_name}_sequences.json")
    imgid2time = save_load_img2time(dataset_path, video_name)
    seqid2time_feat = {}
    for sequence_id, sequence_img_id in action_sequences_user_feats.items():
        seqid2time_feat[sequence_id] = {}
        seqid2time_feat[sequence_id]["start"] = imgid2time[str(sequence_img_id[0])]
        seqid2time_feat[sequence_id]["end"] = imgid2time[str(sequence_img_id[-1])]
        feats = np.load(f"{dataset_path}/action_features/{video_name}/{user_name}/{sequence_id}.npz")
        feats = feats["arr_0"]
        # mean on first axis

        seqid2time_feat[sequence_id]["feat"] = feats
        time_sequence = []
        for i in range(5):
            total_time = float(seqid2time_feat[sequence_id]["end"]) - float(seqid2time_feat[sequence_id]["start"])
            timestep = total_time / 5
            time_feat = float(seqid2time_feat[sequence_id]["start"]) + timestep * (i + 1)
            time_sequence.append(time_feat)
        seqid2time_feat[sequence_id]["time_feats"] = time_sequence
    feat_list = []
    time_list = []
    for seqid in seqid2time_feat.keys():
        for i, time in enumerate(seqid2time_feat[seqid]["time_feats"]):
            feats = seqid2time_feat[seqid]["feat"][0, :, :]
            feats = np.mean(np.array(feats), axis=1)
            feat_list.append(feats.tolist())
            time_list.append(float(time))
    time_list = sorted(time_list)
    return time_list, feat_list


def get_emotion_reco_with_time(dataset_path, user_name, video_name):
    """
    :return: list of time and list of corresponding features
    """
    imgid2feat = read_json_file(
        f"{dataset_path}/emotion/{video_name}/{user_name}_emotion_feat.json")
    imgid2time = save_load_img2time(dataset_path, video_name)
    imageid_list = imgid2feat.keys()
    imageid_list = [int(i) for i in imageid_list]
    imageid_list = sorted(imageid_list)
    emo_feat_time = []
    emo_feat = []
    for imageid in imageid_list:
        emo_feat_time.append(imgid2time[str(imageid)])
        emo_feat.append(imgid2feat[str(imageid)])
    return emo_feat_time, emo_feat


def convert_listfeat_to_df(time_list, data_list, column_prefix):
    """
    convert a list of features inside a dataframe
    :return:
    """
    # get columns for dataframe
    column_names = pd.Series(data_list[0]).to_frame().T.columns.tolist()
    # Create an empty DataFrame
    df = pd.DataFrame(columns=column_names)
    # Iterate over each sublist in the data_list
    for sublist in data_list:
        # Create a new row in the DataFrame with the sublist elements as columns
        row = pd.Series(sublist).to_frame().T
        df = pd.concat([df, row], ignore_index=True)
    # Rename the columns by adding a prefix "action_"
    new_columns = {col: f'{column_prefix}_{col}' for col in df.columns}
    df = df.rename(columns=new_columns)
    df["time"] = time_list
    return df


def df_to_dfblocksized(df, blocksize, aggregation_list, feature_list):
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # Set the "time" column as the index
    df.set_index('time', inplace=True)
    # Resample the DataFrame using the desired time window 'b' and calculate the mean of the features
    dict_agg = {}
    for feat in feature_list:
        dict_agg[feat] = aggregation_list
    df_resampled = df.resample(blocksize).agg(dict_agg)
    # Reset the index if needed
    df_resampled.reset_index(inplace=True)
    return df_resampled


def get_all_feats_per_blocksize(dataset_path, user_name, video_name, openface_feat_names, blocksize, feats):
    """ return a dataframe with merged features """
    pose_feat_time, pose_feat = get_user_headpose(dataset_path, user_name, video_name)
    emo_feat_time, emo_feat = get_emotion_reco_with_time(dataset_path, user_name, video_name)
    action_feat_time, action_feat = get_action_reco_with_time_averaged(dataset_path, user_name, video_name)
    # take dataframe
    dict_df = {}
    if "openface" in feats:
        df_openface = get_user_openface_video_with_time(dataset_path, user_name, video_name)
        dict_df["openface"] = df_to_dfblocksized(df_openface, blocksize, ["mean", "var"], openface_feat_names)
    if "pose" in feats:
        df_pose = convert_listfeat_to_df(pose_feat_time, pose_feat, "pose")
        pose_feat_names = [item for item in df_pose.columns.tolist() if item != 'time']
        dict_df["pose"] = df_to_dfblocksized(df_pose, blocksize, ["mean", "var"], pose_feat_names)
    if "emotion" in feats:
        df_emotion = convert_listfeat_to_df(emo_feat_time, emo_feat, "emotion")
        emotion_feat_names = [item for item in df_emotion.columns.tolist() if item != 'time']
        dict_df["emotion"] = df_to_dfblocksized(df_emotion, blocksize, ["mean", "var"], emotion_feat_names)
    if "action" in feats:
        df_action = convert_listfeat_to_df(action_feat_time, action_feat, "action")
        action_feat_names = [item for item in df_action.columns.tolist() if item != 'time']
        dict_df["action"] = df_to_dfblocksized(df_action, blocksize, ["mean"], action_feat_names)
    merged_df = dict_df[feats[0]]
    if len(feats) > 1:
        for i in range(1, len(feats)):
            merged_df = pd.merge(merged_df, dict_df[feats[i]], on='time', how='outer')
    # merge all dataframes
    # Merge the DataFrames based on the shared column
    # merged_df = df_openface_blocksized
    # merged_df = pd.merge(df_pose_blocksized, df_emotion_blocksized, on='time', how='outer')
    # merged_df = pd.merge(merged_df, df_action_blocksized, on='time', how='outer')
    return merged_df


def convert_segments_videot1_to_videot2(label_segments, video_path_source, video_path_target):
    """ convert video time in ms from video 1 to video 2 time in sec """
    video_source_duration = get_video_duration(video_path_source)
    video_target_duration = get_video_duration(video_path_target)
    for i in range(len(label_segments)):
        label_segments[i]["start"] = ((label_segments[i][
                                           "start"] * 1e-3) * video_target_duration) / video_source_duration
        label_segments[i]["start"] = round(label_segments[i]["start"], 4)
        label_segments[i]["end"] = ((label_segments[i]["end"] * 1e-3) * video_target_duration) / video_source_duration
        label_segments[i]["end"] = round(label_segments[i]["end"], 4)
    return label_segments


def get_df_labels_of_user(dataset_path, user_name, video_name):
    user_id_list = read_json_file(f"{dataset_path}/person_id/{video_name}.json")[user_name]
    elan_path = f'{dataset_path}/labels/tim/{video_name}_processed.eaf'
    eaf = get_elan(elan_path)
    label_segments = get_label_segments(eaf)
    # HERE PROBLEMS OF LABELS IMAGES ARE LONGER THAN EXPECTED EX: 1.30mn against 1mn for real -> interpoler la bonne t
    label_segments = convert_segments_videot1_to_videot2(label_segments,
                                                         f'{dataset_path}/mp4_video/{video_name}_processed.mp4',
                                                         f'{dataset_path}/video/{video_name}.MOV')
    # refine label_segments considering the user
    filtered_label_segments = []
    for segment in label_segments:
        for idx, id in enumerate(segment["id"]):
            if id in user_id_list:
                segment["id"] = id
                segment["iab"] = segment["iab"][idx]
                filtered_label_segments.append(segment)
                break
    return filtered_label_segments


def append_df_feat_to_label(df, label_segments):
    # Create a dictionary to map time to label
    time_to_label = {}
    for window in label_segments:
        start_time = window['start']
        end_time = window['end']
        label = window['iab']  # Assuming each window has only one label value
        time_to_label.update({t: label for t in df['time'] if start_time <= t.timestamp() <= end_time})

    # Add 'label' column to the existing DataFrame
    df['label'] = df['time'].map(time_to_label)
    return df


def load_user_data_df(dataset_path, user_name, video_name, openface_feat_names, blocksize, feat_names):
    df = get_all_feats_per_blocksize(dataset_path, user_name, video_name, openface_feat_names, blocksize, feat_names)
    label_segments = get_df_labels_of_user(dataset_path, user_name, video_name)  # PB DE LABEL SEGMENT
    df = append_df_feat_to_label(df, label_segments)
    # Filter rows with NaN values in the 'label' column
    df = df.dropna(subset=[('label', '')])
    # Specify the columns to exclude from NaN filtering
    exclude_columns = [('label', ''), ('time', '')]
    # Filter rows with NaN values in all other columns
    df = df.dropna(subset=[col for col in df.columns if col not in exclude_columns])
    # check_and_create_folder(f'{dataset_path}/dataset')
    # df.to_csv(f'{dataset_path}/dataset/{video_name}_{user_name}.csv', index=False)
    # Rename the columns using the str.replace() method
    # df = df.rename(columns=lambda x: x.replace('/', '-'))
    return df


def extract_rows_in_window(df, start_time, end_time, feature_columns):
    window_data = df.loc[
        (df['time'] > start_time.iloc[0]) & (df['time'] <= end_time.iloc[0]) & df[feature_columns].notna().all(axis=1)]
    return window_data[feature_columns]


def df_to_numpy_array(df, sequence_time, blocksize):
    feature_columns = [col for col in df.columns if col not in [("time", ""), ("label", "")]]
    df.sort_values('time', inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    training_data = []
    labels = []
    for i, row in df.iterrows():
        if i >= int(sequence_time / blocksize):
            current_time = row['time']
            start_time = current_time - pd.Timedelta(seconds=sequence_time)
            end_time = current_time
            window_data = extract_rows_in_window(df, start_time, end_time, feature_columns)
            if len(window_data) == int(sequence_time / blocksize):
                label = row['label']
                training_data.append(window_data.values.tolist())
                labels.append(label.values.tolist())
    return training_data, labels


def load_numpy_data(dataset_path, user_name, video_name, openface_feat_names, blocksize, sequence_time, feat_names,
                    prefix):
    feat_prefix = ""
    for f in feat_names: feat_prefix += f[0]
    check_and_create_folder(f"{dataset_path}/dataset/{prefix}_{feat_prefix}_{sequence_time}_seq")
    X_path = f"{dataset_path}/dataset/{prefix}_{feat_prefix}_{sequence_time}_seq/{video_name}_{user_name}_X.npy"
    Y_path = f"{dataset_path}/dataset/{prefix}_{feat_prefix}_{sequence_time}_seq/{video_name}_{user_name}_Y.npy"
    if not os.path.isfile(X_path) or not os.path.isfile(Y_path):
        df = load_user_data_df(dataset_path, user_name, video_name, openface_feat_names, blocksize, feat_names)
        training, labels = df_to_numpy_array(df, sequence_time, blocksize=0.5)
        np.save(X_path, np.array(training, dtype=object), allow_pickle=True)
        np.save(Y_path, np.array(labels, dtype=object), allow_pickle=True)
    X = np.load(X_path, allow_pickle=True)
    Y = np.load(Y_path, allow_pickle=True)
    return X, Y


if __name__ == "__main__":
    dataset_path = "../datasets/home_data"
    user_name = "user2"
    video_name = "IMG_0003"
    openface_feat_names = [" gaze_angle_x", " gaze_angle_y", " pose_Rx", " pose_Ry", " pose_Rz"]
    blocksize = "0.5S"
    sequence_time = 2
    # user = get_user_openface_video_with_time(dataset_path, user_name, video_name)
    # get_user_openface_with_time(dataset_path, user_name, video_name)
    # a = get_user_headpose(dataset_path, user_name, video_name)
    # a, b = get_action_reco_with_time_averaged(dataset_path, user_name, video_name)
    # a, b = get_emotion_reco_with_time(dataset_path, user_name, video_name)
    # get_all_feats_per_blocksize(dataset_path, user_name, video_name, openface_feats, "0.5S")
    # get_df_labels_of_user(dataset_path, user_name, video_name)
    # df = load_user_data_df(dataset_path, user_name, video_name, openface_feat_names, blocksize)
    # training, labels = df_to_numpy_array(df, sequence_time, blocksize=0.5)
    a, b = load_numpy_data(dataset_path, user_name, video_name, openface_feat_names, blocksize, sequence_time,
                           prefix_save)
    print("end")
