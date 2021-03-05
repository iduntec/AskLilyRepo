from google_images_download import google_images_download  # importing the library

if __name__ == '__main__':
    response = google_images_download.googleimagesdownload()  # class instantiation
    arguments = {"output_directory": "/home/idan/AskLilyData/Before Validation/dress_shape/",
                 "limit": 1000,
                 "keywords": "Bell dress shape",
                 "print_urls": True,
                 "chromedriver": "/home/idan/PycharmProjects/AskLilyRepo/train_on_gcp/chromedriver"}  # creating list of arguments
    paths = response.download(arguments)  # passing the arguments to the function
    print(paths)  # printing absolute paths of the downloaded images

    # skirts_length:
    # "keywords": "Mini Skirt, Knee Skirt, Midi Skirt, Maxi Skirt, Train Skirt, Asymmetric Skirt",

    # sleeve_length:
    # "keywords": "Short sleeve top,Long sleeve top,Sleeveless top,Off shoulders sleeve top,
    # Strapless top,One Sleeve top",

    # pants_shape:
    # key_list = "Jogger Shaped Pants, Trousers Shaped Pants, Jeans Shaped Pants, Leggings Shaped Pants",

    # dress_shape:
    # key_list = "A-line dress,Shift dress, Kaftan dress, Bodycon dress, Trapeze dress, Bell dress shape ,Wrap dress"

    # dress_waist_line:
    # key_list = "Empire dress waistline ,Waist dress waistline	,Drop dress waistline,Princess cut dress waistline,
    # Without dress waistline"
