# mukham

Mukham (ముఖం; mook-hum) means face in Telugu! mukham is a simple library to detect the largest face in a given image.
The face detection is performed using the DNN algorithm provided by opencv-python. The largest face is simply determined 
by determining the box with largest enclosing area from the bounds of candidate faces detected. 

### Requirements
This package was written and tested in python 3.7+.
You may check the required depencies in `requirements.txt` file.

### Installation
The package can be installed as follows:
```
    pip install mukham
```

### Usage
The package can be used through one of the following ways. 

#### Command-line interface
```
    python -m mukham -i/--input path/to/input_image -o/--ouput path/to/output_largest_face -c/--conf confidence_threshold
```

the `input` argument is required. The `confidence`  is optional, otherwise must be a floating point number with default value of `0.8`.

#### Code
```
    from mukham.detector import detect_largest_face

    bounding_box = detect_largest_face(path_to_input_img, out_path=path_to_output_img, min_conf=confidence_threshold)
```

The `out_path` is an optional keyword argument default to `None`. `min_conf` is also a keyword argument with default set to `0.8`. `detect_largest_face` will return a 2x2 array of the top left and bottom right corner of the bounding box for the largest face.