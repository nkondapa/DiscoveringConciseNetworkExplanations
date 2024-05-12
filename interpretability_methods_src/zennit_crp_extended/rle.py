from label_studio_converter.brush import decode_rle
import numpy as np
import matplotlib.pyplot as plt


def rle2mask(rle, shape):
    """
    Convert run-length encoding to binary mask.
    """
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    # array = np.asarray([int(x) for x in rle.split()])
    array = np.asarray(rle)
    starts = array[0::2][:-1]
    lengths = array[1::2]
    last_end = 0
    print()
    for index, start in enumerate(starts):
        # mask[start:start+lengths[index]] = 1
        curr_start = last_end + start
        print(curr_start, (curr_start+lengths[index]), lengths[index])
        mask[curr_start:(curr_start+lengths[index])] = 1
        last_end = start+lengths[index]

    plt.imshow(mask.reshape(shape).T)
    plt.show()
    return mask.reshape(shape).T


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)