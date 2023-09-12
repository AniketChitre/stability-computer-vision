from objdet.yolov5.detect import run
from classification.trainer import classify
import os
import glob
import shutil

def infer_one(imgpath):
    weights = '/home/Aket95/mysite/objdet/aoi_model/best.pt'    # trained crop detection model weights
    imgsz = (640, 640)  # raw data image size
    conf_thres = 0.90   # required confidence of formulation object detection
    max_det = 1         # maximum 1 formulation sample per image
    save_crop = True

    source = imgpath
    mycroppath = run(weights=weights, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, save_crop=save_crop, source=source)
    print('cropped image path: ', mycroppath)
    result, confidence = classify(mycroppath)
    os.remove(mycroppath)
    return result, round(confidence*100,3)


if __name__ == '__main__':  # allows you to execute code when file runs as a script, but not when imported as a module
    weights = '/Users/ac2349/GitHub/stability-computer-vision/objdet/aoi_model/best.pt'    # trained crop detection model weights
    imgsz = (640, 640)  # raw data image size
    conf_thres = 0.90   # required confidence of formulation object detection
    max_det = 1         # maximum 1 formulation sample per image
    save_crop = True

    # initialise counters for later computing classifier performance metrics
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for imgpath in glob.glob('/Users/ac2349/GitHub/stability-computer-vision/data/test_set/*.png'):
        source = imgpath
        mycroppath = run(weights=weights, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, save_crop=save_crop, source=source)
        print('cropped image path: ', mycroppath)
        result, confidence = classify(mycroppath)
        print('classification result: ', 'Stable' if result else 'Unstable')

        if "True" in imgpath:
            if result == True:
                TP += 1     # True positive
            else:
                FN += 1     # False negative
                print("Error!")     # prints error so we can identify which samples have been misclassified.
        elif "False" in imgpath:
            if result == True:
                FP += 1     # False positive
                print("Error!")
            else:
                TN += 1     # True negative
        else:   # All the files should be labelled with either True or False in their name.
            pass

        # ### Utilising the function just for cropping sample images.
        # SAV_DIR = '/Users/ac2349/GitHub/stability-computer-vision/data/crops'
        # shutil.copy(mycroppath, SAV_DIR + os.path.basename(mycroppath))

        # os.remove(mycroppath)

    # # Performance metrics computed on a batch of images.
    precision = TP/(TP + FP)
    recall    = TP/(TP + FN)

    print(f'precision: {round(precision,3)}')
    print(f'recall: {round(recall,3)}')
    print(f'F1 score: {round(2*precision*recall/(precision + recall), 3)}')