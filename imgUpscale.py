import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def sharpen(img):
    matrix = [[-1, -3, -1], [-3, 25, -3], [-1, -3, -1]]
    kernel = [[elem / 9 for elem in row] for row in matrix]
    kernel = np.array(kernel)
    #alt kernels
    #stronger kernel = np.array([[-1, -3, -1], [-3, 17, -3], [-1, -3, -1]])
    #std kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def unsharpen(img):
    #kernel = [[-0.00391, -0.01563, -0.02344, -0.01563, -0.00391],\
    #          [-0.01563, -0.06250, -0.09375, -0.06250, -0.01563],\
    #          [-0.02344, -0.09375,  1.85938, -0.09375, -0.02344],\
    #          [-0.01563, -0.06250, -0.09375, -0.06250, -0.01563],\
    #          [-0.00391, -0.01563, -0.02344, -0.01563, -0.00391]]
    #kernel = np.array(kernel)
    #return cv2.filter2D(img, -1, kernel)
    gaussian = cv2.GaussianBlur(img, (9,9), 0.0)
    return cv2.addWeighted(img, 2, gaussian, -1, 0, img)


def write_img(img, name):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img_rgb)

def scale_up_img(src):
    #load img and info
    img = cv2.imread(src,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, color = img.shape

    #get a scale up to meet standard, then scale up to keep aspect ratio
    min_height = 1110
    min_width = 816
    
    #pyrup then sharpen a little while we scale up
    img_resize_sharpen = img
    height_img_resize, width_img_resize, scratch = img_resize_sharpen.shape
    count = 0
    while width_img_resize < min_width and height_img_resize < min_height:
        img_resize_sharpen = cv2.pyrUp(img_resize_sharpen)
        img_resize_sharpen = unsharpen(img_resize_sharpen)
        height_img_resize, width_img_resize, scratch = img_resize_sharpen.shape
        count += 1
    
    for i in range(count):
        img_resize_sharpen = unsharpen(img_resize_sharpen)
    
    return img_resize_sharpen


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
    
