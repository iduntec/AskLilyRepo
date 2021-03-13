import os
import base64
import io
import json
import requests
import glob
from handle_data.update_file_names import load_folder_images
import pandas as pd

FILE_NAME = 'file_name'
GT_LABEL = 'gt_label'
IMAGE_PATH = 'path'
DATASET_COLUMNS = [FILE_NAME, GT_LABEL, IMAGE_PATH]

PREDICTED_LABEL = 'Predicted_label'
PREDICTING_MODEL = 'Predicting_model'
CONFIDENCE = 'Confidence'
PREDICTIONS_COLUMNS = [FILE_NAME, PREDICTED_LABEL, CONFIDENCE]

PREDICTIONS = 'predictions'
SCORES = 'scores'
LABELS = 'labels'
PORT = 8080


def container_predict(image_file_path, port_number=8080):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_file_path: Path to a local image for the prediction request.
        image_key: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
    """

    with io.open(image_file_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    file_name = os.path.basename(image_file_path).split(".")[0]
    # The example here only shows prediction with one image. You can extend it
    # to predict with a batch of images indicated by different keys, which can
    # make sure that the responses corresponding to the given image.
    instances = {
        'instances': [
            {'image_bytes': {'b64': str(encoded_image)},
             'key': file_name}
        ]
    }

    # This example shows sending requests in the same server that you start
    # docker containers. If you would like to send requests to other servers,
    # please change localhost to IP of other servers.
    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)

    response = requests.post(url, data=json.dumps(instances))
    # print(response.json())
    return json.loads(response.text)


def extract_response_top_prediction(model_response):
    # returns the top probability prediction and it's label from the model_response to an image

    predictions_dict = model_response.get(PREDICTIONS)[0]

    scores_list = predictions_dict.get(SCORES)
    max_score = max(scores_list)
    max_score_index = scores_list.index(max(scores_list))

    labels_list = predictions_dict.get(LABELS)
    predicted_label = labels_list[max_score_index]

    return predicted_label, round(100 * max_score, 2)


def handle_prediction(response):
    # extracts the max probability prediction and print it
    # returns a list : [predicted_image_key, top_prediction, confidence]
    predicted_image_key = response.get(PREDICTIONS)[0].get('key')
    top_prediction, confidence = extract_response_top_prediction(response)
    print('{} Image was predicted as {} with {}% confidence.'.format(predicted_image_key, top_prediction, confidence))
    return [predicted_image_key, top_prediction, confidence]


def get_dataset_from_category_folder(category_folder_path) -> object:
    dataset_df = pd.DataFrame(columns=DATASET_COLUMNS)
    full_category_folder_path = os.path.join(category_folder_path, '*')

    feature_folder_paths_list = glob.glob(full_category_folder_path)
    if len(feature_folder_paths_list) == 0:
        print('Category Folder has no sub-folders, Exiting.')
        os.sys(exit(1))

    for feature_folder_path in feature_folder_paths_list:
        feature_gt = feature_folder_path.split('/')[-1]
        file_list = glob.glob(os.path.join(feature_folder_path, '*'))
        for file_path in file_list:
            file_name = get_file_name_from_path(file_path)
            dataset_row = pd.DataFrame({FILE_NAME: [file_name], GT_LABEL: [feature_gt], IMAGE_PATH: [file_path]})
            dataset_df = dataset_df.append(dataset_row, ignore_index=True)
    return dataset_df

# def get_class_mapping():


def get_dataset_predictions(input_dataset, port=PORT):
    # Predicts each image in input_dataset using the model on port PORT.
    # returns predictions_df: a dataframe of the form: [FILE_NAME, PREDICTED_LABEL,PREDICTED_CLASS, CONFIDENCE]

    predictions_df = pd.DataFrame(columns=PREDICTIONS_COLUMNS)

    # class_mapping = get_class_mapping()

    for image_path in input_dataset[IMAGE_PATH]:
        network_response = container_predict(image_path, port)
        current_predicted_output_list = handle_prediction(network_response)
        predictions_row = pd.DataFrame({FILE_NAME: [current_predicted_output_list[0]],
                                        PREDICTED_LABEL: [current_predicted_output_list[1]],
                                        # PREDICTED_CLASS:
                                        CONFIDENCE: [current_predicted_output_list[2]]})
        predictions_df = predictions_df.append(predictions_row, ignore_index=True)
    return predictions_df


def get_file_name_from_path(image_path_in):
    return image_path_in.split('/')[-1].split('.')[0]


if __name__ == '__main__':
    # Predict a single image: -----------------------------------------------------------------------------------------
    # image_file_path = '/media/idan/Elements/Validated/skirt_length/Part3_mixed_Google_and_dataset1/check_batches/
    # Knee_Skirt_2.JPEG'
    # network_response = container_predict(image_file_path, port=PORT)
    # handle_prediction(network_response)
    # ------------------------------------------------------------------------------------------------------------------

    # Predict folder images: -------------------------------------------------------------------------------------------
    batch_folder = '/media/idan/Elements/Validated/skirt_length/evaluate_skirt_length_images'
    dataset_df = get_dataset_from_category_folder(batch_folder)
    predicted_df = get_dataset_predictions(dataset_df, port=PORT)
    # ------------------------------------------------------------------------------------------------------------------
