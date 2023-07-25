import os
import numpy as np
import pandas as pd
import math
from torch.utils.data.dataset import Dataset
from imblearn.over_sampling import RandomOverSampler
from tools.utils import read_json_file
from feature_process.feat_synchro import load_numpy_data
import params


class IABDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert len(self.data) == len(self.labels), "Error: data and labels different size"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.data[index], self.labels[index]

def data_flattening(X, Y):
    X_train = []
    Y_train = []
    for video in range(len(X)):
        for sample in range(X[video].shape[0]):
            X_train.append(X[video][sample, :, :])
            Y_train.append(Y[video][sample])
    return X_train, Y_train


def relabel_engage_or_not(Y, threshold=3):
    # relabel
    Y_bis = []
    for i, iab in enumerate(Y):
        if iab <= threshold:
            Y_bis.append([0])
        if iab >= threshold + 1:
            Y_bis.append([1])
        if math.isnan(iab):
            Y_bis.append(Y_bis[i - 1])
    return Y_bis


def get_id_to_normalize(feat_list, openface_list):
    feat_id_list = []
    if len(feat_list) == 1:
        if "openface" in feat_list:
            pass
        if "pose" in feat_list:
            pass
        if "action" in feat_list:
            feat_id_list.append([0, 1024])
        if "emotion" in feat_list:
            feat_id_list.append([0, 20])

    if len(feat_list) == 2:
        if "openface" in feat_list and "emotion" in feat_list:
            feat_id_list.append([len(openface_list) * 2, len(openface_list) * 2 + 20])
        if "openface" in feat_list and "action" in feat_list:
            feat_id_list.append([len(openface_list) * 2, len(openface_list) * 2 + 1024])
        if "pose" in feat_list and "action" in feat_list:
            feat_id_list.append([44, 44 + 1024])
        if "pose" in feat_list and "emotion" in feat_list:
            feat_id_list.append([44, 44 + 20])

    if len(feat_list) == 3:
        if "openface" in feat_list and "emotion" in feat_list and "pose" in feat_list:
            feat_id_list.append([len(openface_list) * 2 + 44, len(openface_list) * 2 + 44 + 20])
        if "openface" in feat_list and "action" in feat_list and "pose" in feat_list:
            feat_id_list.append([len(openface_list) * 2 + 44, len(openface_list) * 2 + 44 + 1024])
    if len(feat_list) == 4:
        feat_id_list.append([len(openface_list) * 2 + 44, len(openface_list) * 2 + 44 + 20])
        feat_id_list.append([len(openface_list) * 2 + 20 + 44, len(openface_list) * 2 + 20 + 44 + 1024])

    return feat_id_list


def normalize_feats(X_train, X_val, id_list):
    for start_feature_id, end_feature_id in id_list:
        # Normalize and rescale the features for the training set
        train_features = X_train[:, :, start_feature_id:end_feature_id]
        train_normalized_features = np.zeros_like(train_features)
        # Normalize and rescale the features for the validation set
        val_features = X_val[:, :, start_feature_id:end_feature_id]
        val_normalized_features = np.zeros_like(val_features)
        for i in range(end_feature_id - start_feature_id):
            feature = train_features[:, :, i]
            min_val = np.min(feature)
            max_val = np.max(feature)
            train_normalized_features[:, :, i] = (feature - min_val) / (max_val - min_val)
            feature_val = val_features[:, :, i]
            val_normalized_features[:, :, i] = (feature_val - min_val) / (max_val - min_val)
        X_train[:, :, start_feature_id:end_feature_id] = train_normalized_features
        X_val[:, :, start_feature_id:end_feature_id] = val_normalized_features

    return X_train, X_val



def get_cleaned_features(dataset_path, val_video, openface_feat_names, blocksize, sequence_time, feat_names,
                         resampling=True):
    video_names = os.listdir(f"{dataset_path}/video")

    # filter test_video
    v = []
    for video in video_names:
        if video not in params.test_videos:
            v.append(video)
    video_names = v

    video_names = [x.split(".")[0] for x in video_names]

    X_t = []
    Y_t = []
    for i, video_name in enumerate(video_names):
        if video_name not in val_video:
            user_name_list = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json").keys()
            for user_name in user_name_list:
                X, Y = load_numpy_data(dataset_path, user_name, video_name, openface_feat_names, blocksize,
                                       sequence_time, feat_names)
                X_t.append(X)
                Y_t.append(Y)
    X_v = []
    Y_v = []
    for video_name in val_video:
        user_name_list = read_json_file(f"{dataset_path}/user_img/{video_name}/user_id2img.json").keys()
        for user_name in user_name_list:
            X, Y = load_numpy_data(dataset_path, user_name, video_name, openface_feat_names, blocksize, sequence_time,
                                   feat_names)
            X_v.append(X)
            Y_v.append(Y)

    # Flatten data
    X_t, Y_t = data_flattening(X_t, Y_t)
    X_v, Y_v = data_flattening(X_v, Y_v)

    Y_t = relabel_engage_or_not(Y_t)
    Y_v = relabel_engage_or_not(Y_v)

    print(f"Train: {len(X_t)}")
    print(f"Val: {len(X_v)}")

    X_t = np.array(X_t).astype('float64')
    Y_t = np.array(Y_t).astype('float64')
    X_v = np.array(X_v).astype('float64')
    Y_v = np.array(Y_v).astype('float64')

    id_feats = get_id_to_normalize(feat_names, openface_feat_names)
    if len(id_feats) > 0:
        X_t, X_v = normalize_feats(X_t, X_v, id_feats)

    # X_v, Y_v = np.array(X_v).astype('float64'), np.array(Y_v).astype('float64')
    X_t = np.nan_to_num(X_t, nan=0.0)
    X_v = np.nan_to_num(X_v, nan=0.0)

    X_t = np.around(X_t, decimals=6)
    X_v = np.around(X_v, decimals=6)

    # Check the new balanced dataset size
    print("Original dataset shape:", X_t.shape, Y_t.shape)

    count = Y_t.sum()
    print(f"Training set: engage: {count} total samples: {Y_t.shape[0]}")
    count = Y_v.sum()
    print(f"Validation set: engage: {count} total samples: {Y_v.shape[0]}")

    # Perform random oversampling
    if resampling:
        # Reshape the features array to 2D if needed
        num_samples, num_rows, num_cols = X_t.shape
        features_2d = X_t.reshape(num_samples, num_rows * num_cols)

        # Perform random oversampling
        oversampler = RandomOverSampler()
        features_resampled, labels_resampled = oversampler.fit_resample(features_2d, Y_t)

        # Reshape the features back to the original shape if needed
        features_resampled = features_resampled.reshape(-1, num_rows, num_cols)

        """
        u = 0
        # check resample has been done in a clean way:
        for i in range(features_resampled.shape[0]):
            for j in range(X_t.shape[0]):
                if np.array_equal(features_resampled[i], X_t[j]):
                    u += 1
        print(f"U: {u}")
        """

        count = labels_resampled.sum()
        print("Resampled dataset shape:", X_t.shape, Y_t.shape)
        print(f"Resampled training set: engage: {count} total samples: {labels_resampled.shape[0]}")
        X_t = features_resampled
        Y_t = labels_resampled

    # Model parameters
    input_dim = X_t[0].shape[-1]
    output_dim = Y_v[0].shape[-1]

    # Create (PyTorch) train & validation data sets
    train_dataset = IABDataset(data=X_t, labels=Y_t)
    validation_dataset = IABDataset(data=X_v, labels=Y_v)
    return train_dataset, validation_dataset, input_dim, output_dim