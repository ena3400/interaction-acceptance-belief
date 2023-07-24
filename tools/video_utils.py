import os

import cv2
import moviepy.editor as moviepy
import numpy as np


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")


def get_video_fps2(video_path):
    clip = moviepy.VideoFileClip(video_path)
    fps = clip.fps
    clip.close()
    return fps


def get_video_fps(video_path):
    # taking the input
    video_capture = cv2.VideoCapture(video_path)
    # get fps
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    return fps


def get_video_duration(video_path):
    clip = moviepy.VideoFileClip(video_path)
    duration = clip.duration
    clip.close()
    return duration


def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def convert_avi_to_mp4(video_path, output_path):
    clip = moviepy.VideoFileClip(video_path)
    video_name = video_path.split("/")[-1].split(".")[0]
    clip.write_videofile(output_path + "/" + video_name + ".mp4")


def save_img(img, output_path):
    cv2.imwrite(output_path, img)


def get_image_and_time(video_path):
    """
    :return: [list of imgs], [video times of imgs]
    """
    total_duration = get_video_duration(video_path + ".MOV")
    frame_rate = get_video_fps(video_path + ".MOV")
    video = cv2.VideoCapture(video_path + ".MOV")
    total_frames = int(round(frame_rate * total_duration))
    t = 0
    img_times = []
    imgs = []
    while True and len(imgs) <= (total_frames):
        success, frame = video.read()
        if not success:
            break
        percent = t * 100 / total_duration
        if round(percent, 1) % 10 == 0:
            print(f"Percent process: {round(percent)}%")
        imgs.append(frame)
        img_times.append(round(t, 5))
        t += 1 / frame_rate
    return imgs, img_times


def save_image_and_time(video_path, output_path):
    """
    :return: [list of imgs], [video times of imgs]
    """
    total_duration = get_video_duration(video_path + ".MOV")
    frame_rate = get_video_fps(video_path + ".MOV")
    video = cv2.VideoCapture(video_path + ".MOV")
    # total_frames = int(round(frame_rate * total_duration))
    t = 0
    img_times = []
    while True:
        success, frame = video.read()
        if not success:
            break
        percent = t * 100 / total_duration
        if round(percent, 1) % 10 == 0:
            print(f"Percent process: {round(percent)}%")
        img_times.append(round(t, 5))
        save_img(frame, output_path + "/" + str(len(img_times)-1) + "_" + str(round(t, 5)) + ".jpg")
        t += 1 / frame_rate


def save_imgs_from_video(video_path, output_folder):
    img_list, img_times = get_image_and_time(video_path)
    create_folder_if_not_exists(output_folder)
    for i, img in enumerate(img_list):
        save_img(img, output_folder + "/" + str(i) + "_" + str(img_times[i]) + ".jpg")


def read_image(image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    if image is not None:
        # print("Image read successfully.")
        return image
    else:
        print("Failed to read the image.")
        return None


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


def add_padding(image, target_ratio):
    """
    add padding on image to targeted ratio
    :param image:
    :param target_ratio:
    :return:
    """
    height, width, _ = image.shape
    current_ratio = width / height

    if current_ratio >= target_ratio:
        # Add vertical padding
        new_height = int(width / target_ratio)
        padding = (new_height - height) // 2
        padded_image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT)
    else:
        # Add horizontal padding
        new_width = int(height * target_ratio)
        padding = (new_width - width) // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT)

    return padded_image


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == "__main__":
    # video_list = os.listdir("../datasets/home_data/video")
    # for video_name in video_list:
    #     video_path = "../datasets/home_data/video/" + video_name
    #     output_folder = "../datasets/home_data/images/" + video_path.split("/")[-1].split(".")[0]
    #     save_imgs_from_video(video_path, output_folder)
    # save_img_from_video(video_path, "datasets/home_data/images/" + video_path.split("/")[-1].split(".")[0])
    print(get_video_fps("../datasets/home_data/mp4_video/IMG_0001_processed.mp4"))
    print(get_video_fps("../datasets/home_data/processed_video/IMG_0001_processed.avi"))
