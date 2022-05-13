import cv2
import numpy as np
import os
import argparse

def sharpen(img):
    #resharpen
    matrix = [[-1, -3, -1], [-3, 25, -3], [-1, -3, -1]]
    kernel = [[elem / 9 for elem in row] for row in matrix]
    kernel = np.array(kernel)
    return cv2.filter2D(img, -1, kernel)


def sharpen_alt(img):
    #second resharpen
    kernel = np.array([[-1, -3, -1], [-3, 17, -3], [-1, -3, -1]])
    return cv2.filter2D(img, -1, kernel)


def sharpen_standard(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def write_img(img, name):
    cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def scale_up_img(src):
    #load img and info
    img_cv2 = cv2.imread(src,1)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    height, width, color = img_cv2.shape

    #get a scale up to meet standard, then scale up to keep aspect ratio
    min_height = 1110
    min_width = 816
    
    #pyrup then sharpen a little while we scale up
    img_cv2_pyrup_sharpen = img_cv2
    height_pyrup, width_pyrup, c_pyrup = img_cv2_pyrup_sharpen.shape

    while width_pyrup < min_width and height_pyrup < min_height:
        img_cv2_pyrup_sharpen = cv2.pyrUp(img_cv2_pyrup_sharpen)
        img_cv2_pyrup_sharpen = sharpen(img_cv2_pyrup_sharpen)
        height_pyrup, width_pyrup, c_pyrup = img_cv2_pyrup_sharpen.shape
    
    #sharpen after?
    return img_cv2_pyrup_sharpen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='/path/to/src/dir/')
    args = parser.parse_args()
    root = args.path
    dest = root + '/scaled_up/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    cards = os.listdir(root)
    for card in cards:
        if card[-4:] != '.png':
            continue
        print('.',end="",flush="True")
        cardPath = root + card
        scaled_card = scale_up_img(cardPath)
        write_img(scaled_card, dest+card)
    print()

if __name__ == '__main__':
    main()
