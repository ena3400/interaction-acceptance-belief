dataset_path = "datasets/iab"
Y_cross_val = [
    [
        "IMG_0037",
        "IMG_0038",
        "IMG_0039",
        "IMG_0040"
    ],
    [
        "IMG_0032",
        "IMG_0034",
        "IMG_0035",
        "IMG_0036"
    ],
    [
        "IMG_0027",
        "IMG_0028",
        "IMG_0030",
        "IMG_0031"
    ],
    [
        "IMG_0021",
        "IMG_0022",
        "IMG_0023",
        "IMG_0024"
    ],
    [
        "IMG_0014",
        "IMG_0015",
        "IMG_0016",
        "IMG_0017",
        "IMG_0018",
        "IMG_0019"
    ]
]
test_videos = [
    "IMG_0147",
    "IMG_0148",
    "IMG_0150",
    "IMG_0153",
    "IMG_0156",
    "IMG_0157",
]
openface_feat_names = [
    " gaze_angle_x",
    " gaze_angle_y",
    " pose_Rx",
    " pose_Ry",
    " pose_Rz",
    " AU01_r",
    " AU02_r",
    " AU04_r",
    " AU05_r",
    " AU06_r",
    " AU07_r",
    " AU09_r",
    " AU10_r",
    " AU12_r",
    " AU14_r",
    " AU15_r",
    " AU17_r",
    " AU20_r",
    " AU23_r",
    " AU25_r",
    " AU26_r",
    " AU45_r"
]

# Data
features = ["openface"]  # list of which can contain openface, pose, action or emotion
blocksize = "0.5S"
sequence_time = 2
resampling = True

# Model
architecture = "IabRnn"
hidden_dims = "128"
lr = 1e-2
batch_size = 512
epochs = 30
weight_decay = 0.
