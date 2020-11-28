import os
import matplotlib.pyplot as plt
import cv2
from google.cloud.vision import types


class AnnotationBox(object):
    """
    This class uses Google's localization_object object
    """

    def __init__(self, image, annotation_name, score, normalized_left_up_dot, normalized_right_down_dot):
        self.input_img = image
        self.annotation_name = annotation_name
        self.score = score
        self.left_up_dot = self.normalized_to_regular_coordinate(normalized_left_up_dot)
        self.right_down_dot = self.normalized_to_regular_coordinate(normalized_right_down_dot)
        self.cropped_box_ndarray = self.crop_box_annotation()
        self.cropped_box_typeImage = self.ndarray_to_image_type()

        # self.complete_output_path = self.create_output_file_path(out_folder)

    def ndarray_to_image_type(self):
        # turn the cropped image ndarray to types.image google object so we can relabel it later on.
        transformed_colors = cv2.cvtColor(self.cropped_box_ndarray, cv2.COLOR_RGB2BGR)
        return types.Image(content=cv2.imencode('.jpg', transformed_colors)[1].tostring())

    def normalized_to_regular_coordinate(self, dot):
        """
        Turns Google output normalized coordinate into regular coordinate using the image.
        :param dot: list [x,y]
        :return:
        """
        pic_height = self.input_img.shape[0]
        pic_width = self.input_img.shape[1]

        [regular_x, regular_y] = [int(pic_width * dot[0]), int(pic_height * dot[1])]
        return [regular_x, regular_y]

    def print_box_annotation_data(self):
        print('Crop annotation: {} ({}%)'.format(self.annotation_name, int(100 * (round(self.score, 2)))))

    def crop_box_annotation(self):
        x1 = self.left_up_dot[0]
        y1 = self.left_up_dot[1]
        x2 = self.right_down_dot[0]
        y2 = self.right_down_dot[1]
        cropped = cv2.cvtColor(self.input_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        return cropped

    # def create_output_file_path(self, out_folder):
    #     box_image_file_name = os.path.basename(self.annotation_name + '.png')
    #     return os.path.join(out_folder, box_image_file_name)

    def save_cropped_box_as_image(self):
        im_bgr = cv2.cvtColor(self.cropped_box_ndarray, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.complete_output_path, im_bgr)


def print_annotations_boxes_dictionary(dictionary):
    print ('Annotations:')
    if not bool(dictionary):
        print ('No annotations detected.')
        print('\n')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(dictionary))
        for box, ax in zip(dictionary.values(), axes):
            box.print_box_annotation_data()

            ax.imshow(box.crop_box_annotation())
            ax.set(title=box.annotation_name)
            ax.axis('off')
        plt.show()
        print('\n')


def save_annotations_boxes_dictionary(dictionary):
    if not bool(dictionary):
        print ('No annotations detected.')
    else:
        for box in dictionary.values():
            box.save_cropped_box_as_image()
