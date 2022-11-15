import time

import cv2
import os
import shutil
import pandas as pd
import os.path
import torch
import warnings
warnings.filterwarnings("ignore")

from Architecture import DenseNet


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def read_and_resize(filename, grayscale=False, fx=1, fy=1):
    # read file
    if grayscale:
        img_result = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    img_result = cv2.resize(img_result, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return img_result


def getImageFromFolderAndMove(pathToImage, pathNewFolderImages):
    # get image and move produced image to another folder
    out = []
    if not os.path.isdir(pathNewFolderImages):
        os.mkdir(pathNewFolderImages)
    if len(os.listdir(pathToImage)) == 0:
        print("Folder toSort is empty!")
    for filename in os.listdir(pathToImage):
        imgPath = (pathToImage + "/").replace("//", "/") + filename
        out = read_and_resize(imgPath)
        shutil.move(imgPath, (pathNewFolderImages + "/").replace("//", "/") + filename)
        return [out, filename]


def resizeImage(img, size):
    # reshape to the shape of (size, size)
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype("float32") / 255


def convertImg(img, device):
    # prepare image to pass it to the model
    return torch.tensor([img.swapaxes(2, 1).swapaxes(1, 0)]).to(device)


def passToModel(model, image):
    return model(image).argmax(1)


def loadModel(pathTo, device):
    model = to_device(DenseNet(), device)
    model.load_state_dict(torch.load(pathTo, map_location=device))
    return model


def logInfo(pathToCsv, imgName, predict, predictClass):
    if not os.path.isfile(pathToCsv):
        df = pd.DataFrame({'name': [],
                           'class': [],
                           'classname': []})
    else:
        df = pd.read_csv(pathToCsv)
    new_row = pd.DataFrame({'name': [imgName],
                            'class': [predict],
                            'classname': [predictClass]})
    df = df.append(new_row, ignore_index=True)
    df.to_csv(pathToCsv, index=False)


def getResizePass(pathToModel, pathToImage, pathNewFolderImages, csvFilePath, classDecode):
    device = get_default_device()
    model = loadModel(pathToModel, device)
    img, filename = getImageFromFolderAndMove(pathToImage, pathNewFolderImages)
    resImg = convertImg(resizeImage(img, 256), device)

    out = passToModel(model, resImg)
    cls = classDecode[int(out[0] + 1)]
    numCls = int(out[0] + 1)
    log = logInfo(csvFilePath, filename, numCls, cls)
    print(f"Model predicted class {numCls}, which is a {cls}")


def main():
    googlePath = "./drive/MyDrive/"
    localPath = "./processing/"
    path = localPath
    pathToModel = path + "model/denceNetModel.pt"  # pretrained model. It takes RGB images with size 256x256
    imgPath = path + "toSort"  # folder with all images to sort
    imgNewPath = path + "sorted"  # folder with all sorted images
    csvPath = path + "logGarbage.csv"  # file for the logging
    classDecoder = {1: "paper/cardboard", 2: "metal", 3: "plastic", 4: "glass"}  # all classes

    start = time.time()
    getResizePass(pathToModel, imgPath, imgNewPath, csvPath, classDecoder)
    end = time.time()
    print(f"Time of execution is {end - start} seconds!")


main()
