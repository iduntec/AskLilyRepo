import os
import base64
import io
import json
import requests
import glob

from handle_data.update_file_names import load_folder_images

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
    predicted_image_key = response.get('predictions')[0].get('key')
    top_prediction, probability = extract_response_top_prediction(response)
    print('{} Image was predicted as {} with {}% confidence.'.format(predicted_image_key, top_prediction, probability))
    return {predicted_image_key: [top_prediction, probability]}


def predict_batch_by_folder(folder_path, port=PORT):
    # loads images from folder_path then predicts each using the model on port PORT.
    # returns predictions_dict: a dictionary where the keys are files names and the values are lists of 2 elements:
    # {<file_name>:[<predicted_class>, <max_probability>]}

    full_folder_path = os.path.join(folder_path, '*')
    batch_paths_list = glob.glob(full_folder_path)
    predictions_dict = {}
    for image_path in batch_paths_list:
        network_response = container_predict(image_path, port)
        tmp_pred_dict = handle_prediction(network_response)
        predictions_dict.update(tmp_pred_dict)
    return predictions_dict


if __name__ == '__main__':
    # Predict a single image: -----------------------------------------------------------------------------------------
    # image_file_path = '/media/idan/Elements/Validated/skirt_length/Part3_mixed_Google_and_dataset1/check_batches/
    # Knee_Skirt_2.JPEG'
    # network_response = container_predict(image_file_path, port=PORT)
    # handle_prediction(network_response)
    # ------------------------------------------------------------------------------------------------------------------

    # Predict folder images: -------------------------------------------------------------------------------------------
    batch_folder = '/media/idan/Elements/Validated/skirt_length/Part3_mixed_Google_and_dataset1/check_batches'
    predict_batch_by_folder(batch_folder, port=PORT)
    # ------------------------------------------------------------------------------------------------------------------
