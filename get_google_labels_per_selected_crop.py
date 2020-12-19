import os
import sys

import pandas as pd

import handle_annotations
import label_and_annotate_by_gcp_vision_api as labeling_funcs

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "C:\Users\Idan\Desktop\AskLily files\My First Project-834fa920a8ad.json"


# get picture from url

# get labels and cropping

# choose wanted crop by category

# send crop to re-labeling

# turn into item's labels-confidence data frame

# concatenate to other item's in the category


class OnlineItem(object):

    def __init__(self, url_in, ask_lily_category_name_in, ask_lily_to_google_mapping_in):

        self.url = url_in
        self.ask_lily_to_google_mapping = ask_lily_to_google_mapping_in
        self.ask_lily_category_name = ask_lily_category_name_in
        self.google_category = ask_lily_to_google_mapping_in.get(ask_lily_category_name_in)
        self.check_input_category()

    def check_input_category(self):
        # checks if ask_lily_category_name_in is a part of the AskLily's categories.
        if self.ask_lily_to_google_mapping.get(self.ask_lily_category_name) is None:
            print('{} is not one of AskLily''s categories, quiting ! '.format(ask_lily_category_name))
            sys.exit()

    def run(self):
        image_object = labeling_funcs.get_web_or_load_local_image(self.url)

        # localize objects: # get localized crops from google:
        localized_crops = labeling_funcs.get_picture_vision_api_object_localization(image_object)

        # translate google's crops object into dictionary:
        self.annotate_boxes_dict = labeling_funcs.localization_crops_object_to_boxes_dict(image_object, localized_crops)

        #display all annotations
        handle_annotations.print_annotations_boxes_dictionary(self.annotate_boxes_dict)

        # choose only relevant annotation:
        self.url_item_annotation_box = self.get_relevant_annotate_box()

        # label relevant annotation and turn into readable dataframe
        self.labels_n_confidences_df = self.re_label_relevant_annotation()

    def re_label_relevant_annotation(self):
        # checks if the requested category found in img, if no retruns empty output table,
        # else - crop and labels annotation, then turns result into a dataframe with
        # the image url, category name, labels and confidences

        category_n_url_headline = self.google_category + ': ' + self.url
        confidence_headline = 'Confidence'
        labels = []
        confidences = []

        if self.url_item_annotation_box is None:  # the requested cat wasn't fount in picture
            output_fd = pd.DataFrame({category_n_url_headline: ['Category was not found in image.'],
                                      confidence_headline: [1]})
            return output_fd[[category_n_url_headline, confidence_headline]]

        else:  # requested category was found in pic -> label relevant annotation only
            self.annotation_labels = labeling_funcs.get_picture_vision_api_labels(
                self.url_item_annotation_box.cropped_box_typeImage)

            for tag in self.annotation_labels:
                labels.append(tag.description)
                confidences.append(round(tag.score, 2))

            output_fd = pd.DataFrame({category_n_url_headline: labels,
                                      confidence_headline: confidences})
        return output_fd[[category_n_url_headline, confidence_headline]]

    def get_relevant_annotate_box(self):
        # pulls only relevant annotation box out of boxes_dict
        relevant_annotation_box = self.annotate_boxes_dict.get(self.google_category)
        if relevant_annotation_box is None:
            print('{} is not one of found annotations in given picture, returning None'.format(self.google_category))

        return relevant_annotation_box


if __name__ == '__main__':
    askLliy_to_google_dict = {'Dresses': 'Dress',
                              'JacketsCoats':'Outerwear',
                              'Tops': 'Top',
                              'Pants': 'Pants'}

    ask_lily_category_name = 'Tops'
    items_url_list = [
        "https://rinazin.com/wp-content/uploads/2020/09/1929010901-2-scaled.jpg"


    ]

    initial_df = pd.DataFrame({})
    for item_url in items_url_list:
        online_item_obj = OnlineItem(item_url, ask_lily_category_name, askLliy_to_google_dict)
        online_item_obj.run()
        tmp_item_output_df = online_item_obj.labels_n_confidences_df
        initial_df = pd.concat([initial_df, tmp_item_output_df], axis=1)

    output_folder = "C:\\Users\\Idan\\Desktop\\AskLily files\\"
    output_file_name = ask_lily_category_name + '_Labels.csv'
    output_file_path = os.path.join(output_folder, output_file_name)

    initial_df.to_csv(output_file_path)
