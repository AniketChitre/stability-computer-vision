
from objdet.yolov5.detect import run
from classification.trainer import classify
import os

if __name__ == '__main__':
    weights = '/Users/ac2349/GitHub/stability-computer-vision/objdet/aoi_model/best.pt'    # crop detection
    imgsz = (640, 640)
    conf_thres = 0.75
    max_det = 1
    save_crop = True
    # source = '/Users/ac2349/GitHub/formulations-imaging/ComputerVision/data/test_set/'
    source = '/Users/ac2349/GitHub/stability-computer-vision/data/images/opencv_01-03-2023_S101_True_post-pHAdj.png'
    
    mycroppath = run(weights=weights, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, save_crop=save_crop, source=source)
    print('cropped image path: ', mycroppath)    
    result = classify(mycroppath)
    print('classification result: ', 'True' if result else 'False')
    os.remove(mycroppath)

    