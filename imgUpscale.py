import cv2
import numpy as np
import os
import argparse
from math import ceil

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

def scale_up_img(img, min_height, min_width):
    #load img info
    height, width, color = img.shape
    img_resize = img
    max_jump = 200

    if height > min_height and width > min_width:
        return img

    #scale up to keep aspect ratio, take steps to get there and sharpen as go
    scale = max(ceil(min_width / width),\
                ceil(min_height / height))
    
    new_width = width * scale
    new_height = height * scale

    diff_width = new_width - width
    diff_height = new_height - height

    num_steps = max(ceil(diff_width / max_jump),\
                    ceil(diff_height / max_jump))
    
    steps = [0]*num_steps
    for i in range(num_steps):
        steps[i] = [width + ceil(diff_width * (i + 1)/num_steps),\
                    height + ceil(diff_height * (i + 1)/num_steps)]

    for i in range(len(steps)):
        size = steps[i]
        img_resize = cv2.resize(img_resize, size) #linear interp
        #sharpen every other to avoid over sharpening
        if i % 2 == 0:
            img_resize = unsharpen(img_resize)
     
    #alternative idea: 
    #    1) resize, gaussBlur, sharpen 4 times
    #    2) pyrup: scale up w/ pyrup (doubles size each time),
    #              sharpen each scale up, then sharpen again once 
    #              for each scale up at the end.
    return img_resize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='/path/to/src/dir/')
    args = parser.parse_args()
    root = args.path
    dest = root + '/scaled_up/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    min_height = 1110
    min_width = 816
    cards = os.listdir(root)
    for card in cards:
        if card[-4:] != '.png':
            continue
        print('.',end="",flush="True")
        card_path = root + card
        img = cv2.imread(card_path,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scaled_card = scale_up_img(img, min_height, min_width)
        write_img(scaled_card, dest+card)
    print()

if __name__ == '__main__':
    main()
    
