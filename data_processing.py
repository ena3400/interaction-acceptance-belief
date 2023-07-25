from tools.video_utils import convert_avi_to_mp4, save_image_and_time
from tools.utils import check_and_create_folder
from feature_process.extract_user_img import save_cropped_img
from feature_process.extract_action_feat import load_model, save_features
from feature_process.extract_emotion_feat import process_video_emotion_feat, load_emonet_model
import os
import cv2


def avi_to_mp4(video_list):
    """
    create mp4 videos
    :return:
    """
    for video_name in video_list:
        video_path = "datasets/home_data/processed_video/" + video_name
        output_folder = "datasets/home_data/mp4_video/"
        convert_avi_to_mp4(video_path, output_folder)


def video_to_images(video_list):
    """
    create images folders
    :return:
    """
    for video_name in video_list:
        print(f'Processing {video_name}')
        video_path = "datasets/home_data/video/" + video_name
        output_folder = "datasets/home_data/images/" + video_path.split("/")[-1].split(".")[0]
        # save_imgs_from_video(video_path, output_folder)
        check_and_create_folder(output_folder)
        save_image_and_time(video_path, output_folder)


def create_user_img(video_list, dataset_path):
    """
    create user_img folders require person_id folders
    :return: folders per user in user img + a json format {"user1:[img_id0, img_id1,...,img_id1230]...}
    """
    check_and_create_folder(f"{dataset_path}/user_img")
    for video_name in video_list:
        video_name = video_name.split(".")[0]
        save_cropped_img(video_name, dataset_path)


def create_action_reco_feats(video_list, dataset_path):
    """
    create action_features folders with a folder per user with features "sequenceid.npz" corresponding to a sequence id
    and a json "userid_sequences.json" per user with corresponding img to sequences id
    :return:
    """
    check_and_create_folder(f"{dataset_path}/action_features")
    model = load_model("feature_process/i3d/models/rgb_charades.pt", dataset="charades")
    # # Check the currently allocated GPU memory
    # allocated_memory = torch.cuda.memory_allocated()
    # print(f"Currently Allocated GPU Memory: {allocated_memory / 1024 ** 2:.2f} MB")
    for video in video_list:
        save_features(video, model, dataset_path)


def create_emotion_feats(video_list, dataset_path):
    """
    create emotion feat files, as json per user
    """
    check_and_create_folder(f"{dataset_path}/emotion")
    model = load_emonet_model('feature_process/emonet/pretrained/emonet_8.pth')
    for video in video_list:
        process_video_emotion_feat(dataset_path, video, model)


if __name__ == "__main__":
    dataset_path = "datasets/home_data/"
    video_list = os.listdir("datasets/home_data/test")
    video_list = [x.split(".")[0] for x in video_list]
    v = []
    for vid in video_list:
        if vid not in []:
            v.append(vid)
    video_list = v
    video_to_images(video_list)
    create_user_img(video_list, dataset_path)
    create_action_reco_feats(video_list, dataset_path)
    create_emotion_feats(video_list, dataset_path)