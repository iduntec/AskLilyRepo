import hashlib
import io
import os
import time

import requests
from PIL import Image
from selenium import webdriver


def fetch_image_urls(query, max_links_to_fetch, wd, sleep_between_interactions):
    def scroll_to_end(wd):
        sleep_between_interactions = 3  # 1 sec
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

        # build the google query

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print("Found: {} search results. Extracting links from {}:{}".format(number_results, results_start,
                                                                             number_results))

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print("Found: {} image links, done!".format(len(image_urls)))
                break
        else:
            print("Found: {}".format(len(image_urls)) + " image links, looking for more ...")
            time.sleep(30)
            # return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path, url):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print("ERROR - Could not download {} - {}".format(url, e))

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print("SUCCESS - saved {} - as {}".format(url, file_path))
    except Exception as e:
        print("ERROR - Could not save {} - {}".format(url, e))


def search_and_download(search_term, driver_path, target_path='./images', number_images=5):
    target_folder = os.path.join(target_path, '_'.join(search_term.lower().split(' ')))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)

    if res is None:
        wd.close()
        return 0

    else:
        for elem in res:
            persist_image(target_folder, elem)
        wd.close()
    return 1


if __name__ == '__main__':
    # feature_name = 'Neck'
    # key_list = ['V neck', 'Sweetheart neck', 'Turtleneck', 'Round neck', 'Boat neck', 'Halter neck',
    #             'Halter strap neck', 'Tailored collar neck', 'straight-across neckline', 'Asymmetric neck']

    feature_name = 'SkirtLength'
    key_list = ['Mini Skirt', 'Knee Skirt', 'Midi Skirt', 'Maxi Skirt', 'Train Skirt', 'Asymmetric Skirt']

    # feature_name = 'SleeveLength'
    # key_list = [ 'Short sleeve top', 'Long sleeve top', 'Sleeveless top', 'Off shoulders sleeve top', 'Strapless top',
    #              'One Sleeve top']

    # feature_name = 'PantsShape'
    # key_list = ['Jogger Shaped Pants', 'Trousers Shaped Pants', 'Jeans Shaped Pants', 'Leggings Shaped Pants']

    output_path = os.path.join('/home/idan/AskLilyData/new_pics', feature_name)
    chrome_driver_path_in = '/home/idan/PycharmProjects/AskLilyRepo/train_on_gcp/chromedriver'
    wanted_number_images = 500
    for key in key_list:
        output_signal = search_and_download(key, chrome_driver_path_in, output_path, wanted_number_images)
