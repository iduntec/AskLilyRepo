from label_and_annotate_by_gcp_vision_api import get_web_or_load_local_image, get_picture_vision_api_labels, \
    print_and_return_labels, get_picture_vision_api_object_localization, localization_crops_object_to_boxes_dict
import pandas as pd
from itertools import repeat
import io
import os
from urlparse import urlparse
import cv2
from google.cloud import vision
from google.cloud.vision import types
import numpy as np
import handle_annotations
import url_opener_class

MAX_HIGH_LEVEL_LABELS_COLUMNS = 5
MAX_DEEP_LEVEL_LABELS_COLUMNS = 10

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "C:\Users\Idan\Desktop\AskLily files\My First Project-834fa920a8ad.json"


def get_high_level_initial_output(askLily_category_name, item_url_s):
    output_df = pd.DataFrame({'AskLily': [askLily_category_name],
                              'URL': [item_url_s]}, index=[0])
    for columns_counter in range(1, MAX_HIGH_LEVEL_LABELS_COLUMNS + 1):
        output_df['Label_' + str(columns_counter)] = [0]
    return output_df


def labels_into_df(localization_object, ask_Lily_category_name, item_url):
    # turns google object into df with :
    # [asklily item category, url, label_1,...label_10]
    output_df = get_high_level_initial_output(ask_Lily_category_name, item_url)

    col_counter = 1
    if not localization_object:
        output_df['Label_' + str(col_counter)] = ['No labels detected.']
        print ('No labels detected.')

    else:
        for box in localization_object:
            if col_counter <= MAX_HIGH_LEVEL_LABELS_COLUMNS:
                output_df['Label_' + str(col_counter)] = [box.name]
                col_counter = col_counter + 1

    return output_df


def add_deep_level_options_to_relevant_high_level(high_level_label, all_options_dict, added_deep_labels,
                                                  added_deep_scores, img_url):
    initial_high_level_dict = all_options_dict[high_level_label]
    for deep_label, deep_score in zip(added_deep_labels, added_deep_scores):
        initial_high_level_dict[deep_label] = [deep_score, img_url]

    all_options_dict[high_level_label] = initial_high_level_dict


def split_deep_level_labels_scores_n_url(temp_content_dict):
    labels = []
    scores = []
    urls = []
    for entry, score_n_url in zip(temp_content_dict.keys(), temp_content_dict.values()):
        labels.append(entry)
        scores.append(score_n_url[0])
        urls.append(score_n_url[1])
    return labels, scores, urls


if __name__ == "__main__":

    # HIGH LEVEL :******************************************************************************************************
    # ask_Lily_category = 'Tops'
    # url_list = ["https://rinazin.com/wp-content/uploads/2020/09/2020066001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/09/1929010901-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/09/2020057311-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/09/2020067001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/09/2020080001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/08/2020064001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/08/2020078001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/08/2020048001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/08/2020083001-2-scaled.jpg",
    #             "https://rinazin.com/wp-content/uploads/2020/08/2020057001-2-scaled.jpg"
    #             ]
    # output_data = pd.DataFrame([])
    #
    # for url in url_list:
    #     image_object = get_web_or_load_local_image(url)
    #
    #     # get labels for whole picture:
    #     localized_crops = get_picture_vision_api_object_localization(image_object)
    #
    #     # labels into df:
    #     tmp_output = labels_into_df(localized_crops, ask_Lily_category, url)
    #
    #     # concat to overall
    #     output_data = output_data.append(tmp_output, ignore_index=True)
    #
    # output_folder = "C:\\Users\\Idan\\Desktop\\AskLily files\\label_options\\"
    # output_file_name = ask_Lily_category + '_Label_options.csv'
    # output_data.to_csv(output_folder + output_file_name)

    # DEEP LEVEL: ******************************************************************************************************
    data_df = pd.read_csv(
        r"C:\Users\Idan\Desktop\AskLily files\label_options\high_level_annotations\high_level_annotation_options.csv")
    url_list = data_df['URL'].to_list()
    url_list = [x for x in url_list if str(x) != 'nan']
    clothing_optional_tags_list = data_df['Unique Relevant Values'].to_list()
    clothing_optional_tags_list = [x for x in clothing_optional_tags_list if str(x) != 'nan']

    # clothing_optional_tags_list = ['Top', 'Outerwear', 'Dress', 'Pants', 'Shorts', 'Shoe', 'Footwear', 'Skirt', 'Tie']

    # create empty dict :
    deep_level_options_dict = {}
    max_set_len = 0
    for option in clothing_optional_tags_list:
        deep_level_options_dict.update({option: {}})

    for url in url_list:
        print(url)
        image_object = get_web_or_load_local_image(url)

        # localize objects:
        localized_crops = get_picture_vision_api_object_localization(image_object)  # get localized crops from google

        # translate google's crops object into dictionary :
        annotate_boxes_dict = localization_crops_object_to_boxes_dict(image_object, localized_crops)

        # get labels for cropped annotations:
        for annotating_box in annotate_boxes_dict.values():
            print('annotating_box : ' + annotating_box.annotation_name)
            if annotating_box.annotation_name in clothing_optional_tags_list:
                print(annotating_box.annotation_name)

                annotation_labels = get_picture_vision_api_labels(annotating_box.cropped_box_typeImage)
                deep_labels, deep_scores = print_and_return_labels(annotation_labels)
                add_deep_level_options_to_relevant_high_level(annotating_box.annotation_name,
                                                              deep_level_options_dict, deep_labels, deep_scores, url)
                max_set_len = max(max_set_len, len(deep_level_options_dict[annotating_box.annotation_name]))

    # pad short dicts with zeros to be a max_set_len size and split urls from deep_labels
    for high_level_option in deep_level_options_dict.keys():
        content_dict = deep_level_options_dict[high_level_option]
        for i in range(0, max_set_len - len(content_dict)):
            content_dict['empty_slot_' + str(i)] = [str(0), str(0)]

        deep_labels_list, deep_labels_scores, deep_labels_urls = split_deep_level_labels_scores_n_url(content_dict)
        deep_level_options_dict[high_level_option] = deep_labels_list
        deep_level_options_dict[high_level_option + '_Label_score'] = deep_labels_scores
        deep_level_options_dict[high_level_option + '_url'] = deep_labels_urls

    deep_level_label_option_df = pd.DataFrame(deep_level_options_dict)
    output_folder = "C:\\Users\\Idan\\Desktop\\AskLily files\\label_options\\deep_level_annotations\\"
    output_file_name = 'deep_Label_options_with_urls.csv'
    deep_level_label_option_df.to_csv(output_folder + output_file_name)
