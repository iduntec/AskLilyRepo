import tkinter as tk
from tkinter import filedialog, Text
import label_and_annotate_by_gcp_vision_api as vision_api_funcs
import os

image_path = []


# Create functionality for buttons
def open_image():
    for widget in frame.winfo_children():
        widget.destroy()

    file_name = filedialog.askopenfile(initialdir="/", title='Select Image')
    image_path.append(file_name)
    print(file_name)


def label_image():
    file_path = image_path.pop()
    # downloaded_image = vision_api_funcs.url_to_image(file_path)
    # image_file_name = os.path.basename(urlparse(input_path).path + '.png')
    #     downloaded_pictures_folder_path = "C:\\Users\\Idan\\Desktop\\AskLily files\\random clothing pictures"
    #     local_image_path = os.path.join(downloaded_pictures_folder_path, image_file_name)
    #     cv2.imwrite(local_image_path, downloaded_image)
    #
    #     # plot image:
    #     # cv2.imshow("Image", downloaded_image)
    #     # cv2.waitKey(0)
    #
    # get labels for whole picture:
    labels_data = vision_api_funcs.get_picture_vision_api_labels_by_path(file_path)
    vision_api_funcs.print_labels(labels_data)


root = tk.Tk()  # instance of the GUI

# GUI background
canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")
canvas.pack()  # attach this to the GUI

# GUI front frame
frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

# Create Buttons:
open_img = tk.Button(root, text="Open Image", padx=10, pady=5, fg="white", bg="#263D42", command=open_image)
open_img.pack()  # attach the file ot the GUI

run_labeling = tk.Button(root, text="Label Image", padx=10, pady=5, fg="white", bg="#263D42", command=label_image)
run_labeling.pack()  # attach the file ot the GUI

root.mainloop()
