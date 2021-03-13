from PIL import Image
import os, os.path
from glob import glob


def load_folder_images(folder_path):
    images_list = []
    for f in os.listdir(folder_path):
        # taking all extensions since there are only images:
        ext = os.path.splitext(f)[1]
        if ext.lower() not in [".jpg", ".gif", ".png", ".tga", "jpeg"]:
            continue

        images_list.append(Image.open(os.path.join(folder_path, f)))

    return images_list


def change_file_list_names(pics_list):
    for file_counter in range(0, len(pics_list)):
        overall_tmp_path = pics_list[file_counter].filename
        file_format = pics_list[file_counter].format
        initial_path = os.path.split(overall_tmp_path)[0]
        folder_name = initial_path.split("/")[-1]
        pics_list[file_counter].filename = os.path.join(initial_path,
                                                        folder_name + "_" + str(file_counter) + "." + file_format)
    return pics_list


def create_new_dir(pics_list, updated_folder_extension):
    current_folder = os.path.split(pics_list[0].filename)[0]
    category_name = current_folder.split("/")[-1]
    father_folder = os.path.split(current_folder)[0]

    output_folder = os.path.join(father_folder, category_name + updated_folder_extension)
    os.mkdir(output_folder)
    return output_folder


def save_file_list(output_directory, pics_list):
    for file_counter in range(0, len(pics_list)):
        file_name = pics_list[file_counter].filename.split("/")[-1]
        pics_list[file_counter].save(os.path.join(output_directory, file_name))


if __name__ == '__main__':
    category_folder_path = "/media/idan/Elements/Validated/skirt_length/Part3_mixed_Google_and_dataset1"
    for feature_folder_path in glob(os.path.join(category_folder_path, "*/")):
        images_list = load_folder_images(feature_folder_path)
        images_list_updated_names = change_file_list_names(images_list)
        output_dir = create_new_dir(images_list_updated_names, "_updated_names")
        save_file_list(output_dir, images_list_updated_names)
