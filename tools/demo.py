#
# The codes are used for implementing CTPN for scene text detection, described in: 
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
# 
# These demo codes (with our trained model) are for text-line detection (without 
# side-refiement part).  
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
# 
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer
#from tesseract_ocr import tesseract_ocr 


DEMO_IMAGE_DIR="demo_images/"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

demo_imnames=os.listdir(DEMO_IMAGE_DIR)
timer=Timer()

timer.tic()

print "demo.py \n caffe path:"
print caffe.__file__

for im_name in demo_imnames:
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #print "Image: %s"%im_name

    im_file=osp.join(DEMO_IMAGE_DIR, im_name)
    im=cv2.imread(im_file)

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    #print len(text_lines)

    #print "Number of the detected text lines: %s"%len(text_lines)

    resImg = "result_tmp/" + im_name
    print resImg

    #im_with_text_lines=draw_boxes(im, text_lines, is_display=False, caption=im_name, wait=False, dstImg = resImg)
    # try:
    #     #pass
    #     tesseract_ocr(im, text_lines, caption=im_name, wait=False, dstImg = im_name)
    # except Exception, e:
    #     print e
        
    
    #im_with_text_lines=draw_boxes(im, text_lines, caption=im_name, wait=False, dstImg = im_file)

#print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#print "Thank you for trying our demo. Press any key to exit..."
print "Time: %f"%(timer.toc() / len(demo_imnames))
# cv2.waitKey(0)
