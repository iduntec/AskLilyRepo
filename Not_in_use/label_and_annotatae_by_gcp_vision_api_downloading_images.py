import io
import os
from urlparse import urlparse
import cv2
from google.cloud import vision
from google.cloud.vision import types
import numpy as np
import handle_annotations
import urlOpenerClass

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


def get_picture_vision_api_labels(img):
    """
    Creates google client, loads a picture from the path 'img_path'
    then asks GCP's labals classifier 'VisionApi' network to label it
    :param img_path: str.
    :return: 'google.protobuf.internal.containers.RepeatedCompositeFieldContainer' object
    """
    client = vision.ImageAnnotatorClient()
    result = client.label_detection(image=img).label_annotations

    return result


def print_labels(RCFC_label_object):
    if not RCFC_label_object:
        print ('No labels detected.')
        print('\n')
    else:
        print('Detected Labels:')
        for d in RCFC_label_object:
            print('Description: {} ({}%)'.format(d.description, int(100 * round(d.score, 2))))
        print('\n')


def get_picture_vision_api_object_localization(imag):
    """
    Creates google client, loads a picture from the path 'img_path'
    then asks GCP's 'VisionApi' for object localization
    :param img_path: str.
    :return: 'google.protobuf.internal.containers.RepeatedCompositeFieldContainer' object
    """
    client = vision.ImageAnnotatorClient()
    result = client.object_localization(image=imag).localized_object_annotations
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
    opener = urlOpenerClass.UrlOpener()
    opener.retrieve('http://www.useragent.org/', 'useragent.html')
    resp = opener.open(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.COLOR_BGR2RGB)

    return image


if __name__ == "__main__":

    input_path = "https://rinazin.com/wp-content/uploads/2020/09/2010058001-2-scaled.jpg"

    # OPTION 1 - use web path :
    image_path = input_path
    # image_file = url_to_image(input_path)
    image_object = get_web_or_load_local_image(image_path)

    # OPTION 2 -  download and save image:
    # image_file = url_to_image(input_path)
    # image_file_name = os.path.basename(urlparse(input_path).path + '.png')
    downloaded_pictures_folder_path = "C:\\Users\\Idan\\Desktop\\AskLily files\\random clothing pictures"
    # image_path = os.path.join(downloaded_pictures_folder_path, image_file_name)
    # cv2.imwrite(local_image_path, image_file)

    # plot image:
    # cv2.imshow("Image", downloaded_image)
    # cv2.waitKey(0)

    # get labels for whole picture:
    labels_data = get_picture_vision_api_labels(image_object)
    print_labels(labels_data)

    # localize objects:
    localized_crops = get_picture_vision_api_object_localization(image_object)  # get localized crops from google

    image_ndarray = url_to_image(image_object.source.image_uri)
    # translate google's crops object into dictionary :
    annotate_boxes_dict = localization_crops_object_to_boxes_dict(image_ndarray, localized_crops,
                                                                  downloaded_pictures_folder_path)

    # print and save annotations as new images
    handle_annotations.print_annotations_boxes_dictionary(annotate_boxes_dict)
    handle_annotations.save_annotations_boxes_dictionary(annotate_boxes_dict)

    # get labels for cropped annotations:
    for annotating_box in annotate_boxes_dict.values():
        annotation_labels = get_picture_vision_api_labels(annotating_box.cropped_box_typeImage)
        print ('Labels for annotation: {}'.format(annotating_box.annotation_name))
        print_labels(annotation_labels)