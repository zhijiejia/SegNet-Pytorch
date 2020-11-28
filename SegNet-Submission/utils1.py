import numpy as np

'''
    1. This file provide the paint function for the output of NetWork
    2. NetWork can predict 32 classes, therefore the length of colormap is 32
'''

colormap = [[64,128,64],[192,0,128],[0,128,192],[0,128,64],[128,0,0],[64,0,128],
           [64,0,192],[192,128,64],[192,192,128],[64,64,128],[128,0,192],[192,0,64],
           [128,128,64],[192,0,192],[128,64,64],[64,192,128],[64,64,0],[128,64,128],
           [128,128,192],[0,0,192],[192,128,128],[128,128,128],[64,128,192],[0,0,64],
           [0,64,64],[192,64,128],[128,128,0],[192,128,192],[64,0,64],[192,192,0],
           [0,0,0],[64,192,0]]

def decode_segmap(label_mask, classes=0):
    if classes==0:
        raise Exception("The classes are illegal!")
    img_height = label_mask.shape[0]
    img_width = label_mask.shape[1]
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, classes):
        r[label_mask == ll] = colormap[ll][0]
        g[label_mask == ll] = colormap[ll][1]
        b[label_mask == ll] = colormap[ll][2]
    rgb = np.zeros((img_height, img_width, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)