import numpy as np
from scipy import fft
import cv2
from operator import itemgetter


def read_img(img_path):
    """Reading input image

    Args:
        img_path (str): path of the image

    Returns:
        img (numpy.array): original image in the form of numpy array
        original_image (numpy.ndarray): orignal image
        overlay (numpy.ndarray): copy of orignal image
        width (int): width of the image
        height (int): height of the image
    """
    image = img_path
    original_image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.imread(image, 0)

    overlay = original_image.copy()

    img = np.array(image)
    height, width = img.shape

    return img, original_image, overlay, width, height

def create_quantize_dct(img, width, height, block_size, stride, Q_8x8):
    """Following are the actions performed in this function:
        a) Create sliding windows
        b) Apply dct transform to each block
        c) Quantize all dct coefficients

    Args:
        img (numpy.array): original image
        width (int): width of the image
        height (int): height of the image
        block_size (int): the block size of matrix
        stride (int): Sliding window stride count / overlap

    Returns:
        quant_row_matrices (list): quantized blocks as rows
    """
    quant_row_matrices = []

    for i in range(0, height - block_size, stride):
        for j in range(0, width - block_size, stride):
            block = img[i: i + block_size, j: j + block_size]
            
            # dct
            dct_matrix = fft.dct(block)

            # quantization of dct co-effs
            quant_block = np.round(np.divide(dct_matrix, Q_8x8))
            block_row = list(quant_block.flatten())

            # left-corner pixel co-ordinates and block
            quant_row_matrices.append([(i, j), block_row])
    
    return quant_row_matrices


def lexographic_sort(quant_row_matrices):
    """Lexicographic sort. Following are the operations performed:
        a)Finding matched blocks
        b)Euclidean operations for calculating shift vectors

    Args:
        quant_row_matrices (list): quantized blocks as rows

    Returns:
        shift_vec_count(list): The count of shift vectors
        matched_blocks (dict): Dictionary of the blocks matched
    """
    sorted_blocks = sorted(quant_row_matrices, key=itemgetter(1))

    # FORMAT: [[block1], [block2], (pos1), (pos2), shift vector]
    matched_blocks = []

    # to keep track of sf count
    shift_vec_count = {}

    for i in range(len(sorted_blocks) - 1):
        if sorted_blocks[i][1] == sorted_blocks[i + 1][1]:
            point1 = sorted_blocks[i][0]
            point2 = sorted_blocks[i + 1][0]

            # shift vector
            s = np.linalg.norm(np.array(point1) - np.array(point2))

            # increment count for s
            shift_vec_count[s] = shift_vec_count.get(s, 0) + 1
            matched_blocks.append([sorted_blocks[i][1], sorted_blocks[i + 1][1],
                                point1, point2, s])
    
    return shift_vec_count, matched_blocks


def shift_vector_thresh(shift_vec_count, matched_blocks, shift_thresh):
    """Applying the shift vector threshold

    Args:
        shift_vec_count (list): The count of shift vectors
        matched_blocks (dict): Dictionary of the blocks matched
        shift_thresh (int): Shift threshold

    Returns:
        matched_pixels_start (list): list of all the matched pixels by shift vector threshold
    """
    matched_pixels_start = []
    for sf in shift_vec_count:
        if shift_vec_count[sf] > shift_thresh:
            for row in matched_blocks:
                if sf == row[4]:
                    matched_pixels_start.append([row[2], row[3]])
    
    return matched_pixels_start


def display_results(overlay, original_image, matched_pixels_start, block_size):
    """Displaying results

    Args:
        overlay (numpy.ndarray): the overlay image
        original_image (numpy.ndarray): the original image
        matched_pixels_start (list[list]): list of all the matched pixels by shift vector threshold
        block_size (int): the block size of matrix
    """
    alpha = 0.5
    orig = original_image.copy()

    for starting_points in matched_pixels_start:
        p1 = starting_points[0]
        p2 = starting_points[1]

        overlay[p1[0]: p1[0] + block_size, p1[1]: p1[1] + block_size] = (0, 0, 255)
        overlay[p2[0]: p2[0] + block_size, p2[1]: p2[1] + block_size] = (0, 255, 0)

    cv2.addWeighted(overlay, alpha, original_image, 1, 0, original_image)
    
    cv2.imshow("Original Image", orig)
    cv2.imshow("Detected Forged/Duplicated Regions", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
