import os
import cv2
import dlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from tools.video_utils import *
from tools.utils import read_json_file, check_and_create_folder, save_json
from tools.bbox_utils import convert_bbox_minmax_wh, extract_image_bbox, calculate_bbox_surface

from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from torchvision import transforms


def transform_img(img):
    plt.imshow(img)
    plt.show()
    img = cv2.resize(img, (512, 512))
    transform_fn = transforms.Compose([
        video.VideoToTensor(),
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_list = transform_fn([img])
    return img_list


def predict_action(img_path):
    img = read_image(img_path)
    img_list = transform_img(img)
    net = get_model('inceptionv3_ucf101', nclass=101, pretrained=True)
    pred = net(nd.array(img_list[0]).expand_dims(axis=0))
    classes = net.classes
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input video frame is classified to be')
    for i in range(topK):
        print('\t[%s], with probability %.3f.' %
              (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))


def save_img_user_cropped(img_path, json_user_img2id):
    data_path = img_path.split("images")[0]
    video_name = img_path.split("/")[-2]
    img_name = img_path.split("/")[-1]
    img_id = int(img_name.split("_")[0])
    img = read_image(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get user identities
    user_identities = read_json_file(f"datasets/home_data/person_id/{video_name}.json")
    img = letterbox(img, (640), stride=64, auto=True)[0]
    # get bbox value of img
    bbox_per_frames = read_json_file(data_path + "tracking/" + video_name + "_processed.json")
    for id in range(len(bbox_per_frames["data"][img_id])):
        # filter reflected bbox
        if calculate_bbox_surface(*bbox_per_frames["data"][img_id][id]["bbox"]) > 10000:
            bbox_minmax = bbox_per_frames["data"][img_id][id]["bbox"]
            tracker_id = bbox_per_frames["data"][img_id][id]["id"]
            kp = bbox_per_frames["data"][img_id][id]["kp"]
            # select the good user_name
            for k, v in user_identities.items():
                if tracker_id in v:
                    user_name = k

            if 'user_name' not in locals():
                print(f"error on {tracker_id}, {img_path}")
            else:
                # convert bbox and crop the img
                bbox_wh = convert_bbox_minmax_wh(*bbox_minmax)
                bbox_wh = [int(i) for i in bbox_wh]
                # img = draw_bounding_box(img, *bbox_wh)
                cropped = extract_image_bbox(img, *bbox_wh, 5, 30)
                cropped = add_padding(cropped, 5/3)
                # do not save if there is not the head
                if kp[2] >= 0.90:
                    json_user_img2id[user_name].append(img_id)
                    cv2.imwrite(f"datasets/home_data/user_img/{video_name}/{user_name}/{len(json_user_img2id[user_name])-1}.png", cropped)
    return json_user_img2id
    # plt.imshow(cropped)
    # plt.plot()
    # print(bbox_minmax)
    # print(bbox_per_frames["data"][img_id][0])


def save_cropped_img(video_name, dataset_path):
    """
        create user_img folders require person_id folders
        :return: folders per user in user img + a json format {"user1:[img_id0, img_id1,...,img_id1230]...}
    """
    print(f"Processing {video_name}")
    # get user identities and create folders
    user_identities = read_json_file(f"{dataset_path}/person_id/{video_name}.json")
    # initialize json to save
    json_user_img2id = {}
    for user_n in user_identities.keys():
        check_and_create_folder(f"{dataset_path}/user_img/{video_name}")
        check_and_create_folder(f"{dataset_path}/user_img/{video_name}/{user_n}")
        json_user_img2id[user_n] = []
    img_list = os.listdir(f"{dataset_path}/images/{video_name}")
    sorted_images = sorted(img_list, key=lambda x: int(x.split('_')[0]))
    for img_name in sorted_images:
        img_path = f"{dataset_path}/images/{video_name}/{img_name}"
        json_user_img2id = save_img_user_cropped(img_path, json_user_img2id)
    save_json(json_user_img2id, f"{dataset_path}/user_img/{video_name}/user_id2img.json")


def save_face_user_cropped(img_path, json_user_img2id, predictor, detector):
    video_name = img_path.split("/")[-2]
    img_name = img_path.split("/")[-1]
    user_img_id = int(img_name.split("_")[0])
    user_id2img = read_json_file(f"../datasets/home_data/user_img/{video_name}/user_id2img.json")
    id = user_id2img[user_img_id]
    user_identities = read_json_file(f"../datasets/home_data/person_id/{video_name}.json")
    # read the image
    img = cv2.imread(img_path)
    # Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    id = 0
    for i, face in enumerate(faces):
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point
        # Crop the face
        cropped_face = img[y1:y2, x1:x2]
        # Resize the cropped face to 224x224
        resized_face = cv2.resize(cropped_face, (224, 224))
        # Save the cropped face
        json_user_img2id[user_identities].append(id)
        destination_path = f'../datasets/home_data/openface_img/{video_name}/{user_identities}'
        cv2.imwrite(f"{destination_path}/{len(os.listdir(destination_path))}", resized_face)
    return json_user_img2id


def save_cropped_face_img(video_name):
    """
        create user_img folders require person_id folders
        :return: folders per user in user img + a json format {"user1:[img_id0, img_id1,...,img_id1230]...}
    """
    print(f"Processing {video_name}")
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # get user identities and create folders
    user_identities = read_json_file(f"../datasets/home_data/person_id/{video_name}.json")
    # initialize json to save
    json_user_img2id = {}
    for user_n in user_identities.keys():
        check_and_create_folder(f"../datasets/home_data/openface_img/{video_name}")
        check_and_create_folder(f"../datasets/home_data/openface_img/{video_name}/{user_n}")
        json_user_img2id[user_n] = []

    img_list = os.listdir(f"../datasets/home_data/images/{video_name}")
    sorted_images = sorted(img_list, key=lambda x: int(x.split('_')[0]))
    for img_name in sorted_images:
        for user in user_identities:
            img_path = f"../datasets/home_data/user_img/{video_name}/{user}/{img_name}"
            json_user_img2id = save_face_user_cropped(img_path, json_user_img2id, predictor, detector)
    save_json(json_user_img2id, f"../datasets/home_data/openface_img/{video_name}/user_id2img.json")


if __name__ == "__main__":
    video_list = os.listdir("../datasets/home_data/video")
    already_done = os.listdir("../datasets/home_data/user_img")
    for video_name in video_list:
        video_name = video_name.split(".")[0]
        if video_name not in already_done:
            # predict_action(img_list)
            # save_img_user_cropped(img_path)
            save_cropped_img(video_name)
    #save_cropped_img("IMG_0027")
        # TO DO LISTER LES VIDEO AVEC REFLET SUR ECRAN ET FILTER LES BBOX QUI EN RESSORTE GRAVE A LEUR TAILLE.
