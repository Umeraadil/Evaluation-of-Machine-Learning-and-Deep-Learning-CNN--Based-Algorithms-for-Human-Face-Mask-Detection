"""
Author: Gulnawaz Gani
"""
import os
import cv2
import dlib
from cv2 import namedWindow, WINDOW_AUTOSIZE, imshow
import glob
import matplotlib.pyplot as plt


def get_boundingbox(face, width, height, scale=1.0, minsize=None):  # scale=1.3i
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def face_preprocessing(img, fname):
    # Face detector
    face_detector = dlib.get_frontal_face_detector()
    print(face_detector)

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Detect with dlib
    image = img  # cv2.imread('C:\\Users\\HP\\PycharmProjects\\deepfakes\\a5.jpg')
    # Image size
    height, width = image.shape[:2]
    print(f' height = {height} and width = {width}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    print(f' height = {height} and width = {width}')
    # cv2.imshow("test", gray)
    faces = face_detector(gray)
    print(len(faces))
    print("len(faces)")
    if len(faces) == 0:
        return
    if len(faces):
        # For now only take biggest face
        face = faces[0]

        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        print(x, y, size)
        cropped_face = image[y:y + size, x:x + size]
        # cut down height by 25%
        height, width = cropped_face.shape[:2]
        lower_img = cropped_face[int(1/5 * 224):-1, :, :]


        # Text and bb
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        #label = 'face'
        color = (0, 255, 0)
        #cv2.putText(image, str() + label, (x, y + h + 30),
                    #font_face, font_scale,
                    #color, thickness, 2)
        # draw box over face
        #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Show
    namedWindow("test", WINDOW_AUTOSIZE)
    # imshow('test', image)
    # imshow('test', cropped_face)
    path = r''
    cv2.imwrite(os.path.join(path, "C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/DATASET/cropped datasets/crop color dataset 4/withoutmask/" + fname[:-4] + ".jpg"), lower_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f'image size={image.shape}')


if __name__ == '__main__':
    imdir = 'C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/DATASET/Train/Non Mask/'
    ext = ['png', 'jpg', 'jpeg']  # Add image formats here

    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv2.imread(file) for file in files]
    for i in range(len(images)):
        fname = os.path.basename(files[i])
        face_preprocessing(images[i], fname)
        # imshow('test', images[i])
        # cv2.waitKey(100)
        # print(files)
        # imshow('test', images[1])
        # print(len(images))