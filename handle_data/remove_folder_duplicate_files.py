import os
from glob import glob
from handle_data.update_file_names import load_folder_images, save_file_list, create_new_dir

def choose_unique_image(images_list_in):
    images_dict = {}
    for image in images_list_in:
        file_name_with_ext = image.filename.split("/")[-1].split('.')[0]
        file_name_with_index = file_name_with_ext.split('.')[0]
        file_name = file_name_with_index[:-2]
        images_dict[file_name] = image
    return images_dict


def change_file_dict_names(pics_dict):
    for image_name, image_file in zip(pics_dict.keys(), pics_dict.values()):
        overall_tmp_path = image_file.filename
        file_format = image_file.format
        initial_path = os.path.split(overall_tmp_path)[0]
        pics_dict.get(image_name).filename = os.path.join(initial_path, image_name + "." + file_format)
    return pics_dict


if __name__ == '__main__':

    # this script takes a category folder like: /home/idan/Downloads/Micha_s_images/TopShape/
    #  that has feature sub-folders like:       /home/idan/Downloads/Micha_s_images/TopShape/T shape
    # that holds duplicated files with different index counter of the form :
    # galita_4832_0.jpg, galita_4832_1.jpg, galita_4832_2.jpg and saves the first file to a new folder named:
    # /home/idan/Downloads/Micha_s_images/TopShape/T shape unique images

    folder_path = "/home/idan/Downloads/Micha_s_images/TopShape/"
    for feature_folder_path in glob(os.path.join(folder_path, "*/")):
        images_list = load_folder_images(feature_folder_path)
        unique_images_dict = choose_unique_image(images_list)
        updated_dict = change_file_dict_names(unique_images_dict)
        output_dir = create_new_dir(images_list, ' unique images')
        save_file_list(output_dir, list(updated_dict.values()))
