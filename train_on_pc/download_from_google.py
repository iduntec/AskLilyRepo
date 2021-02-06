from google_images_download import google_images_download  # importing the library

if __name__ == '__main__':
    response = google_images_download.googleimagesdownload()  # class instantiation
    arguments = {"output_directory": "/home/idan/AskLilyData/Before Validation/",
                 "limit": 1000,
                 "keywords": "Short sleeve top,Long sleeve top,Sleeveless top,Off shoulders sleeve top,Strapless top,"
                             "One Sleeve top",
                 "print_urls": True,
                 "chromedriver": "/home/idan/PycharmProjects/AskLilyRepo/train_on_gcp/chromedriver"}  # creating list of arguments
    paths = response.download(arguments)  # passing the arguments to the function
    print(paths)  # printing absolute paths of the downloaded images


    # SkirtsLength:
    # "keywords": "Mini Skirt, Knee Skirt, Midi Skirt, Maxi Skirt, Train Skirt, Asymmetric Skirt",

    # SleeveLength:
    # "keywords": "Short sleeve top,Long sleeve top,Sleeveless top,Off shoulders sleeve top,
    # Strapless top,One Sleeve top",

    # PantsShape:
    # key_list = "Jogger Shaped Pants, Trousers Shaped Pants, Jeans Shaped Pants, Leggings Shaped Pants",
