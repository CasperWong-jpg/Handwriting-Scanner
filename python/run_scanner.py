import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from scanner_functions import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    im_height, im_width, _ = im1.shape
    bboxes, bw = findLetters(im1)

    # Q5.3
    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    # Q5.4
    bboxes_by_row = dict()
    for (minr, minc, maxr, maxc) in bboxes:
        centerR, centerC = int((maxr + minr) / 2), int((maxc + minc) / 2)
        nearbyR = {centerR + i for i in range(int(-im_height / 12), int(im_height / 12))}
        intersectingR = nearbyR.intersection(bboxes_by_row.keys())
        if len(intersectingR) > 0:
            # A row already exists. We can add the letter to this row
            assert (len(intersectingR) == 1)
            existingR = intersectingR.pop()
            assert (bboxes_by_row[existingR].get(centerC, None) is None)
            bboxes_by_row[existingR][centerC] = (minr, minc, maxr, maxc)
        else:
            # A row does not exist. Create one!
            bboxes_by_row[centerR] = {centerC: (minr, minc, maxr, maxc)}
    sorted_rows = sorted(bboxes_by_row.keys())

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('nn_weights.pickle','rb'))

    res = ""
    for row in sorted_rows:
        sorted_cols = sorted(bboxes_by_row[row].keys())
        last_maxc = sorted_cols[0]
        for col in sorted_cols:
            minr, minc, maxr, maxc = bboxes_by_row[row][col]
            # Extract letter from image
            letter_img = bw[minr:maxr, minc: maxc]
            # Pad the image and inverse colors
            letter_img = np.pad(letter_img, (int((maxr - minr) / 5), int((maxc - minc) / 5)))
            # Resize and dilate
            letter_img = skimage.morphology.dilation(skimage.transform.resize(letter_img, (32, 32)),
                                                     skimage.morphology.square(2))
            # Transpose and flatten
            letter_img = letter_img.T.flatten()[None]
            # Invert colors
            letter_img = 1 - letter_img

            # Check if we need a whitespace
            diff = minc - last_maxc
            if diff > 1.25 * (maxc - minc):
                res += " "

            # Run neural network to see what alphanumeric letter this is
            h1 = forward(letter_img, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            letter_idx = np.argmax(probs)
            res += letter_list[letter_idx]

            # For debugging
            plt.imshow(letter_img.reshape(32, 32).T)
            # plt.show()

            # Save lastcol for next loop
            last_maxc = maxc

        res += "\n"
    print(res)