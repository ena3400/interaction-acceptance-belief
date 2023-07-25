import torch
from torchvision.models.video import r3d_18
import cv2
import numpy as np
import os
from i3d.pytorch_i3d import InceptionI3d


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def infer_i3d(input, mode="flow"):
    # (batch, channel, t, h, w) with 16 frames clip in input,
    # 9 minimum, correspond to 0.64s
    # features = i3d.extract_features(inputs)
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        load_model = "i3d/models/flow_charades.pt"
    else:
        i3d = InceptionI3d(400, in_channels=3)
        load_model = "i3d/models/rgb_charades.pt"
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    i3d.train(False)
    features = i3d.extract_features(input.cuda())
    return features


def infer_r3d_torch(input):
    # (batch, channel, t, h, w) -> (batch,400 feats)
    model = r3d_18(pretrained=True).cuda()
    features = model(input.cuda())
    return features


if __name__ == "__main__":
    inputs = torch.zeros((1, 3, 128, 224, 224)).cuda()
    # print(features.shape)
    model = r3d_18(pretrained=True).cuda()
    outputs = model(inputs)
    print(outputs.shape)
