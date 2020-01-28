import numpy as np
import cv2
import dlib

# set environment variable
import os
os.environ['OPENCV_IO_ENABLE_JASPER']= 'TRUE'


class DimensionError(Exception):
    """
        raised when the image does not meet the required 
        maximum dimensions of 1024 x 1024.
    """
    def __init__(self, h, w):
        message = "Image is too big " + str((h, w))
        message += "; max allowed size is (1024, 1024)"
        super(DimensionError, self).__init__(message)


def hog_face_detector(img):
    """
        Face detection using the HoG face detection algorithm
        from dlib library.

        Parameters
        ----------
        img: numpy.ndarray achieved from reading an image using
        cv2.imread()

        Returns
        -------
        bounding box: array of two points (x,y):
        top left and bottom right
    """
    detector = dlib.get_frontal_face_detector()
    faces, scores, idx = detector.run(img, 1, -1)

    # check if any faces exist
    assert len(faces) > 0, "No faces found!"

    # get the largest face from the faces detected
    biggest_face_id = 0
    max_size = [faces[0].height(), faces[0].width()]
    for id, face in enumerate(faces[1:]):
        size = [face.height(), face.width()]

        if size[0] * size[1] > max_size[0] * max_size[1]:
            max_size = size
            biggest_face_id = id

    # return the largest bounding box
    tl = faces[biggest_face_id].tl_corner()
    h, w = max_size
    return [(tl.x, tl.y), (tl.x+w, tl.y+h)]


def dnn_face_detector(img):
    """
        Face detection using the DNN face detection algorithm
        from cv2 library.

        Parameters
        ----------
        img: numpy.ndarray achieved from reading an image using
        cv2.imread()

        Returns
        -------
        bounding box: array of two points (x,y):
        top left and bottom right
    """
    # get dimensions of image
    h, w = img.shape[:2]

    # detect faces using DNN algorithm from cv2
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    rgb_mean = np.mean(img, axis=(0, 1)) # mean rgb values to remove effects of illumination
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300, 300), rgb_mean)
    net.setInput(blob)
    faces = net.forward()

    # check if any faces exist
    assert faces.shape[2] > 0, "No faces found!"

    # get the largest face
    first_face = 0 # let first face be biggest face
    biggest_box = faces[0, 0, first_face, 3:7] * np.array([w, h, w, h])
    sx, sy, ex, ey = biggest_box.astype("int")
    max_conf = max([faces[0, 0, i, 2] for i in range(faces.shape[2])])
    biggest_conf = faces[0, 0, first_face, 2]
    biggest_confxsize = (ex - sx)*(ey - sy)*biggest_conf / h*w*max_conf

    for i in range(1, faces.shape[2]):
        # check for erroneous box bounds: must not be greater than 1
        check_bounds = list(map(lambda x: x > 1, faces[0, 0, i, 3:7]))
        if any(check_bounds):
            continue

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        conf = faces[0, 0, i, 2]
        confxsize = (endX - startX)*(endY - startY)*conf / h*w*max_conf

        if confxsize > biggest_confxsize:
            sx, sy, ex, ey = startX, startY, endX, endY
            biggest_confxsize = confxsize

    # return the largest bounding box
    return [(sx, sy), (ex, ey)]


def haar_face_detector(img):
    """
        Face detection using the Haar Cascades algorithm
        from cv2 library.

        Parameters
        ----------
        img: numpy.ndarray achieved from reading an image using
        cv2.imread()

        Returns
        -------
        bounding box: array of two points (x,y):
        top left and bottom right
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Algorithm
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    assert len(faces) > 0, "No faces found!"

    # find the largest image
    max_x, max_y, max_w, max_h = faces[0]
    for (x, y, w, h) in faces[1:]:
        if w*h > max_w*max_h:
            max_x, max_y, max_w, max_h = x, y, w, h
    
    # return the largest bounding box
    return [(max_x, max_y), (max_x+max_w, max_y+max_h)]


def detect_largest_face(in_path, out_path):
    """
        detects the largest face for a given image using Haar
        Cascades algorithm from cv2 library

        Parameters
        ----------
        in_path: path to the input image file
        out_path: path to save output file

        Returns
        -------
        None
    """
    img = cv2.imread(in_path)
    
    # check image dimensions
    h, w = img.shape[:2]
    if h > 1024 or w > 1024:
        raise DimensionError(h, w)

    # detect largest face
    tl, br = hog_face_detector(img)
    largest_face_crop = img[tl[1]:br[1], tl[0]:br[0]]

    # save face image to the given output path
    cv2.imwrite(out_path, largest_face_crop)