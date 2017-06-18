import cv2, caffe
import numpy as np
from matplotlib import cm
import Image
import pytesseract


class foo:
  def __init__(self, n=0):
    self.n = n
  def __call__(self, i):
    self.n += i
    return self.n


def tesseract_ocr(im, bboxes, is_display=True, color=None, caption="Image", wait=True, dstImg = "result.jpg"):
    """
        boxes: bounding boxes
    """
    im=im.copy()
    a = 0
    
    for box in bboxes:
        #print im.size
        #print box[0],box[1],box[2],box[3]
        #print box[2] - box[0], box[3] - box[1]
        if (box[2] - box[0]) > 7 * (box[3] - box[1]):
            roi = im[box[1]: box[3],box[0]: box[2]]
            a= a + 1
            name = "tmp/" + dstImg[:-4] + str(a) + dstImg[-4:]
            cv2.imwrite(name, roi)
            image = Image.fromarray(roi)
            #help(pytesseract.image_to_string)
            vcode = pytesseract.image_to_string(image, config='nobatch digits')
            print vcode




