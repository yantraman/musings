import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image


def show(img):
    temp = np.swapaxes(img, 0, 2)
    fixed_img = np.swapaxes(temp, 0, 1)
    plt.imshow(fixed_img)
    plt.show()


def read_image(mydir, width=32):
    img = Image.open(mydir).convert('RGB')
    img = img.resize((width, width), PIL.Image.ANTIALIAS)
    return np.array(img).reshape((width, width, 3))


def load():
    # getting the ratings
    rating_dirs = glob.glob("./hot_or_not_image_and_rating_data/female/*.txt")
    rating_dirs += glob.glob("./hot_or_not_image_and_rating_data/male/*.txt")
    ratings = np.array([float(open(myDir).read().split('\n')[0]) for myDir in rating_dirs])

    img_dirs = glob.glob("./hot_or_not_image_and_rating_data/female/*.jpg")
    img_dirs += glob.glob("./hot_or_not_image_and_rating_data/male/*.jpg")
    images = np.array([read_image(myDir) for myDir in img_dirs])
    images = images.astype('float32')
    images /= 255
    return images, ratings
