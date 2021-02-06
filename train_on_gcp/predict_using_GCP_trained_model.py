import base64
import io
import json
import requests


def container_predict(image_file_path, image_key, port_number=8080):
    with io.open(image_file_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # The example here only shows prediction with one image. You can extend it
    # to predict with a batch of images indicated by different keys, which can
    # make sure that the responses corresponding to the given image.
    instances = {
        'instances': [
                {'image_bytes': {'b64': str(encoded_image)},
                 'key': image_key}
        ]
    }

    # This example shows sending requests in the same server that you start
    # docker containers. If you would like to send requests to other servers,
    # please change localhost to IP of other servers.
    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)

    response = requests.post(url, data=json.dumps(instances))
    print(response.json())


if __name__ == '__main__':
    image_file_path = '/home/idan/Desktop/hh.jpg'
    image_key = 'first'
    container_predict(image_file_path, image_key, port_number=8080)
