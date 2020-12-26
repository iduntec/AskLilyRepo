from selenium import webdriver
import time
import requests
import shutil
import os
import argparse
import numpy as np


# import cv2
# from vision_api import url_opener_class
#
#
# def url_to_image(url):
#     # download the image, convert it to a NumPy array, and then read
#     # it into OpenCV format
#     opener = url_opener_class.UrlOpener()
#     opener.retrieve('http://www.useragent.org/', 'useragent.html')
#     resp = opener.open(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.COLOR_BGR2RGB)
#     opener.close()
#     return image

def save_img(inp, img, i, directory):
    try:
        filename = inp + str(i) + '.jpg'
        response = requests.get(img, stream=True)
        image_path = os.path.join(directory, filename)
        with open(image_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
            return 1
    except Exception:
        return 0


def find_urls(inp, url, driver, directory, max_img_amount):
    driver.get(url)
    for _ in range(500):
        driver.execute_script("window.scrollBy(0,10000)")
        try:
            driver.find_element_by_css_selector('.mye4qd').click()
        except:
            continue
    successful_downloads = 0
    for j, imgurl in enumerate(driver.find_elements_by_xpath('//img[contains(@class,"rg_i Q4LuWd")]')):
        try:
            imgurl.click()
            img = driver.find_element_by_xpath(
                '//body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute(
                "src")
            was_save_successful = save_img(inp, img, j, directory)
            time.sleep(1)
            successful_downloads += was_save_successful
            if successful_downloads == max_img_amount:
                return

        except:
            pass


def download_and_save_google_images(search_key, output_path, chrome_driver_path_in, wanted_pic_amount):
    parser = argparse.ArgumentParser(description='Scrape Google images')
    parser.add_argument('-s', '--search', default=search_key, type=str, help='search term')
    # parser.add_argument('-d', '--directory', default='../Downloads/', type=str, help='save directory')
    args = parser.parse_args()
    driver = webdriver.Chrome(chrome_driver_path_in)
    directory = output_path + '\\' + search_key
    inp = args.search
    if not os.path.isdir(directory):
        os.makedirs(directory)
    url = 'https://www.google.com/search?q=' + str(
        inp) + '&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947'
    find_urls(inp, url, driver, directory, wanted_pic_amount)


if __name__ == "__main__":
    # key_list = ['V neck', 'Sweetheart neck', 'Turtleneck', 'Round neck', 'Boat neck', 'Halter neck',
    #             'Halter strap neck', 'Tailored collar neck', 'straight-across neckline', 'Asymmetric neck']

    key_list = [ 'Halter neck top']

    output_path = 'C:\\Users\\Idan\\Desktop\\AskLily files\\AutoML'
    chrome_driver_path_in = 'chromedriver.exe'
    max_pics_to_download = 50
    for key in key_list:
        download_and_save_google_images(key, output_path, chrome_driver_path_in, max_pics_to_download)
