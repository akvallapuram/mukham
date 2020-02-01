import numpy as np
import cv2
import errno

# set environment variable
import os
os.environ['OPENCV_IO_ENABLE_JASPER']= 'TRUE'   # allows JPEG2000 format

# path of this file
det_path = os.path.split(os.path.abspath(__file__))[0] + '/'


class DimensionError(Exception):
    """
        raised when the image does not meet the required 
        maximum dimensions of 1024 x 1024.
    """
    def __init__(self, h, w):
        message = "Image is too big " + str((h, w))
        message += "; max allowed size is (1024, 1024)"
        super(DimensionError, self).__init__(message)


def detect_largest_face(in_path, out_path=None, min_conf=0.8):
    """
        Detects largest face using the DNN face detection algorithm
        from cv2 library.

        Args:
            in_path (str): path of the input image file
            out_path (str, optional): path of the cropped image file. Defaults to None
            min_conf (float, optional): threshold confidence to detect face. Defaults to 0.8
        
        Returns:
            bounding_box: an 2x2 array of two (x,y) coordinates 
                for top left and bottom right of the bounding box
    """

    # check input file path
    if not os.path.isfile(in_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), in_path)
    
    # check output file path
    if out_path:
        try:
            with open(out_path, 'w') as f:
                pass
        
        except OSError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_path)

    # read file
    img = cv2.imread(in_path)

    # check image dimensions
    h, w = img.shape[:2]
    if h > 1024 or w > 1024:
        raise DimensionError(h, w)

    # detect faces using DNN algorithm from cv2
    net = cv2.dnn.readNetFromCaffe(
        det_path + "models/deploy.prototxt",
        det_path + "models/res10_300x300_ssd_iter_140000.caffemodel"
        )

    rgb_mean = np.mean(img, axis=(0, 1)) # mean rgb values to remove effects of illumination
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300, 300), rgb_mean)
    net.setInput(blob)
    faces = net.forward()

    # get the most confident faces
    conf_faces = np.array(list(filter(lambda x: x[2] > min_conf, faces[0, 0])))

    # check if any faces exist
    assert len(conf_faces) > 0, "No faces found!"

    # get the largest face
    first_face = 0 # let first face be biggest face
    first_box = conf_faces[first_face, 3:7] * np.array([w, h, w, h])
    sx, sy, ex, ey = first_box.astype("int")

    for i in range(1, conf_faces.shape[0]):
        box = conf_faces[i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        if (endX - startX)*(endY - startY) > (ex - sx)*(ey - sy):
            sx, sy, ex, ey = startX, startY, endX, endY

    # save the crop
    if out_path:
        largest_crop = img[sy:ey, sx:ex]
        saved_file = cv2.imwrite(out_path, largest_crop)
    
    # return the largest bounding box
    bounding_box = [(sx, sy), (ex, ey)]
    return bounding_box