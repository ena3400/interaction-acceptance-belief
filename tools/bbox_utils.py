import cv2


def get_iou(bb1, bb2, convert=False):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # if convert:
    #     bb1 = convert_bbox_dict(bb1)
    #     bb2 = convert_bbox_dict(bb2)
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def convert_bbox_list_to_dict(bbox):
    """
    take as input bbox list
    [x1,y1,x2,y2] or [x_min,y_min,x_max,y_max]
    return bbox dict {x1,y1,x2,y2}
    """
    bbox_dict = {}
    bbox_dict['x1'] = bbox[0]
    bbox_dict['y1'] = bbox[1]
    bbox_dict['x2'] = bbox[2]
    bbox_dict['y2'] = bbox[3]
    return bbox_dict


def convert_bbox_minmax_wh(x_min, y_min, x_max, y_max):
    x = x_min
    y = y_min
    width = x_max - x_min
    height = y_max - y_min
    return x, y, width, height


def calculate_bbox_surface(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    surface = width * height
    return surface


def extract_image_bbox(image, bbox_x, bbox_y, bbox_width, bbox_height, pourcentage_hauteur, pourcentage_largeur):
    height, width, _ = image.shape

    # Calcul de l'augmentation de taille en fonction des pourcentages
    augmentation_ratio_hauteur = 1 + pourcentage_hauteur / 100
    augmentation_ratio_largeur = 1 + pourcentage_largeur / 100

    # Calcul des nouvelles dimensions de la bounding box
    new_width = int(bbox_width * augmentation_ratio_largeur)
    new_height = int(bbox_height * augmentation_ratio_hauteur)

    # Calcul des nouvelles coordonnées de la bounding box
    new_x = int(bbox_x - (new_width - bbox_width) / 2)
    new_y = int(bbox_y - (new_height - bbox_height) / 2)

    # Calcul du padding nécessaire si les nouvelles dimensions dépassent les dimensions de l'image
    padding_x = max(0, -new_x, new_x + new_width - width)
    padding_y = max(0, -new_y, new_y + new_height - height)

    # Mise à jour des nouvelles coordonnées de la bounding box en tenant compte du padding
    new_x += padding_x
    new_y += padding_y

    # Augmentation de la taille de l'image avec le padding
    enlarged_image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

    # Extraction de la région de la bounding box agrandie
    enlarged_bbox = enlarged_image[new_y:new_y + new_height, new_x:new_x + new_width]

    return enlarged_bbox


def draw_bounding_box(image, bbox_x, bbox_y, bbox_width, bbox_height, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), color, thickness)
    return image


def normalize_keypoints(keypoints, target_width=1.0, target_height=1.0):
    # Step 2: Identify bounding box of keypoints
    x_coordinates = [keypoints[i] for i in range(0, len(keypoints), 2)]
    y_coordinates = [keypoints[i] for i in range(1, len(keypoints), 2)]
    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    # Step 3: Calculate scaling factors
    width = max_x - min_x
    height = max_y - min_y
    x_scaling_factor = target_width / width
    y_scaling_factor = target_height / height

    # Step 4: Normalize keypoints
    normalized_keypoints = []
    for i in range(0, len(keypoints), 2):
        x = keypoints[i]
        y = keypoints[i + 1]
        normalized_x = (x - min_x) * x_scaling_factor
        normalized_y = (y - min_y) * y_scaling_factor
        normalized_keypoints.extend([normalized_x, normalized_y])

    return normalized_keypoints


def project_keypoints(keypoints, input_width, input_height, output_width, output_height):
    # Calculate the scaling factors for keypoints
    x_scale = output_width / input_width
    y_scale = output_height / input_height
    # Resize and project the keypoints to the output frame
    resized_keypoints = keypoints.copy()
    resized_keypoints[0] *= x_scale  # Resize x-coordinates
    resized_keypoints[1] *= y_scale
    return resized_keypoints


def is_point_in_bbox(point, bbox):
    """
    :param point: [x,y]
    :param bbox: [x_min,y_min,x_max,y_max]
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox

    if x_min <= x <= x_max and y_min <= y <= y_max:
        return True
    else:
        return False
