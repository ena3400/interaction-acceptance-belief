import json
import os


def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)


def read_json_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def get_file_name(file_name):
    return file_name.split("/")[-1].split(".")[0]


def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder '{folder_path}' has been created.")
    # else:
    #     print(f"The folder '{folder_path}' already exists.")


def is_ordered(lst):
    # Check for ascending order
    if all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)):
        return True

    # Check for descending order
    if all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1)):
        return True

    # If neither ascending nor descending order
    return False


if __name__ == "__main__":
    bbox_paths = os.listdir(f"../datasets/home_data/bbox/")
    print(bbox_paths[0])
    a = read_json_file(f"datasets/home_data/bbox/" + bbox_paths[0])
    print(a["data"][2][0])
