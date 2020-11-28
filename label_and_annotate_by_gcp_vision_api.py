import io
import os
from urlparse import urlparse
import cv2
from google.cloud import vision
from google.cloud.vision import types
import urllib
import numpy as np
import handle_annotations

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "C:\Users\Idan\Desktop\AskLily files\My First Project-834fa920a8ad.json"


# ImageAnnotatorClient doc : https://googleapis.dev/python/vision/1.0.0/gapic/v1/api.html


def get_web_or_load_local_image(picture_path):
    if picture_path.startswith('http') or picture_path.startswith('gs:'):
        image = types.Image()
        image.source.image_uri = picture_path
    else:
        with io.open(picture_path, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)

    return image


def get_picture_vision_api_labels_by_path(img_url_path):
    """
    Creates google client, loads a picture from the path 'img_path'
    then asks GCP's labals classifier 'VisionApi' network to label it
    :param img_path: str.
    :return: 'google.protobuf.internal.containers.RepeatedCompositeFieldContainer' object
    """
    client = vision.ImageAnnotatorClient()
    img = get_web_or_load_local_image(img_url_path)
    result = client.label_detection(image=img).label_annotations

    return result


def print_labels(RCFC_label_object):

    if not RCFC_label_object:
        print ('No labels detected.')
        print('\n')
    else:
        print('Detected Labels:')
        for d in RCFC_label_object:
            print('Description: {} ({}%)'.format(d.description, int(100*round(d.score, 2))))
        print('\n')


def get_picture_vision_api_object_localization_by_path(img_path):
    """
    Creates google client, loads a picture from the path 'img_path'
    then asks GCP's 'VisionApi' for object localization
    :param img_path: str.
    :return: 'google.protobuf.internal.containers.RepeatedCompositeFieldContainer' object
    """
    client = vision.ImageAnnotatorClient()
    img = get_web_or_load_local_image(img_path)
    result = client.object_localization(image=img).localized_object_annotations
    return result


def localization_crops_object_to_boxes_dict(complete_img, localization_object, output_fold):
    """
    turns the google data structure into a dictionary of AnnotationBox(es)
    :param localization_object:
    :return: dictionary holding all the annotation boxes and their data
    """
    boxes_dict = {}
    for box in localization_object:
        box_annotation = box.name
        box_score = box.score

        box_left_up = box.bounding_poly.normalized_vertices[0]
        box_right_down = box.bounding_poly.normalized_vertices[2]

        box_left_up_coordinate = [box_left_up.x, box_left_up.y]
        box_right_down_coordinate = [box_right_down.x, box_right_down.y]

        tmp_box = handle_annotations.AnnotationBox(complete_img, box_annotation, box_score, box_left_up_coordinate,
                                                   box_right_down_coordinate, output_fold)
        boxes_dict.update({box.name: tmp_box})

    return boxes_dict


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    # Only made by Whirldata
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.COLOR_BGR2RGB)

    return image


if __name__ == "__main__":

    input_path = "https://images.squarespace-cdn.com/content/v1/5442b6cce4b0cf00d1a3bef2/" \
                 "1599590792838-52XZCN97YTGA8U99E4MJ/ke17ZwdGBToddI8pDm48kH008e-LTJnP7TSvtOWhULRZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJF" \
                 "bgE-7XRK3dMEBRBhUpxu9lu6X1UF4ibVjKnfq5mTFPExLuuto8jSLtBAasYGVPSSwB4kKIvl69GnqYlJW7U/Ethical-Sustainable-Clothing-People-Tree"
    # input_path = "https://i.dailymail.co.uk/i/pix/2013/01/18/article-2263988-16FED98D000005DC-73_634x731.jpg"

    # download and save image:
    downloaded_image = url_to_image(input_path)
    image_file_name = os.path.basename(urlparse(input_path).path + '.png')
    downloaded_pictures_folder_path = "C:\\Users\\Idan\\Desktop\\AskLily files\\random clothing pictures"
    local_image_path = os.path.join(downloaded_pictures_folder_path, image_file_name)
    cv2.imwrite(local_image_path, downloaded_image)

    # plot image:
    # cv2.imshow("Image", downloaded_image)
    # cv2.waitKey(0)

    # get labels for whole picture:
    labels_data = get_picture_vision_api_labels_by_path(local_image_path)
    print_labels(labels_data)

    # localize objects:
    localized_crops = get_picture_vision_api_object_localization_by_path(local_image_path)  # get localized crops from google

    # translate google's crops object into dictionary :
    annotate_boxes_dict = localization_crops_object_to_boxes_dict(downloaded_image, localized_crops,
                                                                  downloaded_pictures_folder_path)

    # print and save annotations as new images
    handle_annotations.print_annotations_boxes_dictionary(annotate_boxes_dict)
    handle_annotations.save_annotations_boxes_dictionary(annotate_boxes_dict)

    # get labels for cropped annotations:
    for annotating_box in annotate_boxes_dict.values():
        annotation_labels = get_picture_vision_api_labels_by_path(annotating_box.complete_output_path)
        print ('Labels for annotation: {}'.format(annotating_box.annotation_name))
        print_labels(annotation_labels)




