import numpy as np
import skimage


letter_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S',
               'T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']


def overlap_rect(bot1, left1, top1, right1, bot2, left2, top2, right2, epsilon=3) -> bool:
    """
    Check if two rectangles overlap each other
    Cite: https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    """
    # If one rectangle is on left side of other
    if left1 - right2 > epsilon or left2 - right1 > epsilon:
        return False

    # If one rectangle is above other
    if bot1 - top2 > epsilon or bot2 - top1 > epsilon:
        return False

    return True


# takes a color image
# returns a list of bounding boxes and binary black_and_white image (threshold)
def findLetters(image):
    bboxes = []
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # estimate noise and denoise
    original = skimage.img_as_float(image)
    sigma = 0.05
    noisy = skimage.util.random_noise(original, var=sigma**2)
    denoised_image = skimage.restoration.denoise_wavelet(noisy, channel_axis=-1, rescale_sigma=True)

    # greyscale image
    denoised_image = skimage.color.rgb2gray(denoised_image)

    # apply threshold
    thresh = skimage.filters.threshold_otsu(denoised_image)
    # apply morphology
    morph_res = skimage.morphology.closing(denoised_image < thresh, skimage.morphology.square(1))

    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(morph_res)

    # label image regions
    label_image = skimage.measure.label(cleared)

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            # Check if this is a redundant box
            for (otr_minr, otr_minc, otr_maxr, otr_maxc) in bboxes:
                if overlap_rect(minr, minc, maxr, maxc, otr_minr, otr_minc, otr_maxr, otr_maxc):
                    # Two bounding boxes boxing same letter
                    # Pop conflicting box (watch out for shifting indexes) and join to make the largest bounds
                    bboxes.remove((otr_minr, otr_minc, otr_maxr, otr_maxc))
                    minr, minc = min(minr, otr_minr), min(minc, otr_minc)
                    maxr, maxc = max(maxr, otr_maxr), max(maxc, otr_maxc)
            bboxes.append((minr, minc, maxr, maxc))

    return bboxes, morph_res.astype(float)
