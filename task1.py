"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import sys

import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
            :param test_img:
            :param characters:

    """
    # TODO Add your code here. Do not modify the return and input arguments

    # enrollment()
    dict_chr = {}  # imply a dictionary for characters, key_point: extracted key points from sift, index: what character is
    img_array = np.copy(test_img)
    for i in characters:
        key_points, index = enrollment(test_img, i)
        dict_chr[index] = key_points

    # detection()
    blocks = detection(test_img)

    result = recognition(img_array, blocks, dict_chr, test_img)

    return result
    # raise NotImplementedError


def enrollment(test_img, character):
    """ Args:
        You are free to decide the input arguments.
        images in characters files, upper threshold, lower threshold
    Returns:
    You are free to decide the return.
    sift edge detection drawing out keypoint and feature description
    """

    # TODO: Step 1 : Your Enrollment code should go here.

    index, char_image = character
    (thresh, char_image) = cv2.threshold(char_image, 100, 255, cv2.THRESH_BINARY)

    sift = cv2.SIFT_create()
    keypoint, descriptors = sift.detectAndCompute(char_image, None)

    img = cv2.drawKeypoints(char_image, keypoint, test_img)

    filename = 'sift_keypoint_extract' + index + '.jpg'
    cv2.imwrite(filename, img)
    return descriptors, index


def neighbours(node, test_image):
    # Connected Component Analysis,Connected Component Labeling
    i, j = node
    size = test_image.shape
    w = size[0]
    h = size[1]
    neighbour = [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]
    res = []
    # ans = set(res)
    for pix in neighbour:
        pix_w, pix_h = pix
        if pix_w in range(w) and pix_h in range(h) and test_image[pix_w][pix_h] < 0.5:
            # find black pixels represent characters that have detected
            res.append(pix)
    return set(res)


def draw_blocks(connect, image):
    # block = []
    result = []
    tmp = []
    for pix in connect:
        min_y = min(pix, key=lambda t: t[1])
        max_y = max(pix, key=lambda t: t[1])
        min_x = min(pix, key=lambda t: t[0])
        max_x = max(pix, key=lambda t: t[0])
        start = (min_y[1], min_x[0])
        end = (max_y[1], max_x[0])
        block = []
        block.append(start)
        block.append(end)
        # tmp.append(block)
        result.append(block)
        image = cv2.rectangle(image, start, end, (0, 0, 0), 1)

    show_image(image, delay=1000)
    return image, result


def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.


    Drawing boxes in test images on characters
    Calculating histograms of gradient orientations (8 reference angles) for all pixels inside each sub-patch

    """

    # TODO: Step 2 : Your Detection code should go here.

    w, h = test_img.shape
    # w = size[0]
    # h = size[1]
    (threshold, test_img) = cv2.threshold(test_img, 230, 255, cv2.THRESH_BINARY)
    visited_pix = np.array(np.zeros([w, h]))
    connect = []
    for i in range(w):  # row
        for j in range(h):  # column
            if test_img[i][j] < 0.5 and visited_pix[i][j] != 1:  # in test_img_ rgb = 0 part and traversed pixels
                result = []
                visited = set()
                nodes = set([(i, j)])

                while nodes:
                    node = nodes.pop()
                    x, y = node
                    visited_pix[x][y] = 1
                    visited.add(node)
                    nodes |= neighbours(node, test_img) - visited
                    # print('nodes',nodes)
                    result.append(node)
                connect.append(result)

    (block_image, block) = draw_blocks(connect, test_img)
    filename = 'blocks_test_image' + '.jpg'
    cv2.imwrite(filename, block_image)

    # print(block)
    # print(block_image)

    return block


def recognition(block_img, blocks, characters_list, test_img):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.

    extracting features from block box and test_image
    identifying the most similar pairs using brute forced
    recognition using SIFT
    """
    # TODO: Step 3 : Your Recognition code should go here.
    res = []

    for ele in blocks:
        # print('ele', ele)
        start, end = ele
        x_min, y_min = start
        x_max, y_max = end
        extract_features = block_img[y_min:y_max, x_min:x_max]
        (thresh, extract_features) = cv2.threshold(extract_features, 230, 255, cv2.THRESH_BINARY)
        try:

            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(extract_features, None)
            
        except:
            pass

        match_res = ssd_match(extract_features, descriptors, characters_list)
        match_dic = {'bbox': [x_min, y_min, x_max - x_min, y_max - y_min], 'name': match_res}
        res.append(match_dic)
    return res


def ssd_match(test_img, descriptors, character_dict):
    dist = 165

    if descriptors is None:
        d_ans = 'UNKNOWN'
    else:
        d_ans = 'UNKNOWN'
        for character in character_dict.keys():
            # print('char:', char)
            match_value = []
            character_value = character_dict[character]
            if character is None:
                return 'UNKNOWN'
            else:

                for i in range(len(descriptors)):
                    # print('des:', des)
                    min_dist = sys.maxsize
                    match_ssd = 0
                    for j in range(len(character_value)):
                        if ssd(descriptors[i], character_value[j]) < min_dist:
                            min_dist = ssd(descriptors[i], character_value[j])
                            match_ssd = j

                            match_value.append(match_ssd)
                    # for m in match_value:
                    if min_dist < dist:
                        dist = min_dist
                        d_ans = character
    return d_ans


def ssd(A, B):
    squares = (A - B) ** 2
    return np.sqrt(np.sum(squares))


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
