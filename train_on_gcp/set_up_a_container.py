import os


def mount_model_to_container(container_name, port, local_model_path, sudo_password):
    os.environ["Local_Model_Path"] = local_model_path
    os.environ["Container_Model_Path"] = "/tmp/mounted_model/0001"

    os.environ["CPU_DOCKER_GCR_PATH"] = "gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest"
    os.environ["CONTAINER_NAME"] = container_name
    os.environ["PORT"] = str(port)

    pull_command = 'docker pull ${CPU_DOCKER_GCR_PATH}'
    os.popen("sudo -S %s" % pull_command, 'w').write(sudo_password)

    mount_command = 'docker run --rm --name ${CONTAINER_NAME} -p ${PORT}:8501 ' \
                    '-v ${Local_Model_Path}:${Container_Model_Path} -t ${CPU_DOCKER_GCR_PATH}'
    os.popen("sudo -S %s" % mount_command, 'w').write(sudo_password)


if __name__ == '__main__':
    contain_name = 'skirts_len'
    model_port = 8080
    model_path = '/home/idan/tmp/mounted_model/0001/'
    sudo_pass = 'eee123rrr'

    mount_model_to_container(contain_name, model_port, model_path, sudo_pass)
