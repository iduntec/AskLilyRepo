import os, io
from google.cloud import vision
from google.cloud.vision import types
import cv2
import os
import glob

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "C:\Users\Idan\Desktop\AskLily files\My First Project-834fa920a8ad.json"


def starter(batch):
    client = vision.ImageAnnotatorClient()
    for path in batch:
        if path.startswith('http') or path.startswith('gs:'):
            image = types.Image()
            image.source.image_uri = path
        else:
            with io.open(path, 'rb') as image_file:
                content = image_file.read()
            image = types.Image(content=content)

    result = client.label_detection(image=image).label_annotations

    return result


from pathlib import Path

imgb = []
img_dir = Path('C:\Users\Idan\Desktop\AskLily files\Ex_Files_Azure_Architects_Design_Strategy\Exercise Files\Images')

# get imags 'windowspath'
for i in img_dir.glob('*.jpg'):
    imgb.append(i)

# get images name strings
img_str=[]
for i in imgb:
    img_str.append(str(i))

print(img_str)

data_obj = starter(img_str)
print (data_obj)

# data_path = os.path.join(img_dir, '*g')
# files = glob.glob(data_path)
# data = []
# for f1 in files:
#     img = cv2.imread(f1)
#     data.append(img)
