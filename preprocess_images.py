""" functions to preprocess the images downloaded by the fetch_data_facial_scorer script
    ideally the dataset should only contain a single front face on  every image
    it will be necessary to delete any images that objectively don't belong to the objective of the dataset
    i.e.: strong side views of faces, obviously photoshopped images that distort the face, male faces in a female face dataset, etc...
"""

import sys, os, glob, re
import cv2, PIL
from PIL import Image

def preprocess(haarcascades_dir, images_folder=u'images', faces_folder=u'faces'):
    """ preprocess all the images present in the "images" folder

    Parameters
    ----------
    haarcascades_dir : string, mandatory
        path to opencv haarcascades
    images_folder : string, optional, default u'images'
        path to the folder containing the downloaded images
    faces_folder : string, optional, default u'faces'
        path to the folder containing the faces cropped from the original images
    """

    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)
    frontalface_path = os.path.join(haarcascades_dir, 'haarcascade_frontalface_alt.xml')
    images_filename = os.path.join(images_folder, '*.jpg')
    images = glob.glob(images_filename)
    images = map(lambda x: x[x.rfind("\\")+1:-4], images)
    num_img_with_faces = 0
    for img_name in images:
        print img_name
        img_dir = os.path.join(images_folder, img_name + '.jpg')
        img = cv2.imread(img_dir)
        cascade = cv2.CascadeClassifier(frontalface_path)
        # optimal parameters are scaleFactor=1.05, minNeighbors=3, minSize=(30,30),
        # although minNeighbors can comfortably range between 3-6
        # this configuration will yield better face recognition by minimizing false positives,
        # but at the cost of losing some faces (false negatives)
        # another option is to be less strict by decreasing minNeighbors to 2 and manually
        # deleting the face images generated, so that each original image only has one preprocess face image
        # i.e.: images that don't contain a face, contain multiple faces, etc...
        rects = cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30,30))
        if len(rects) == 0:
            print "No face detected!"
            continue
        rects[:, 2:] += rects[:, :2]
        num_faces = 1
        for x1, y1, x2, y2 in rects:
            cut = img[y1:y2, x1:x2]
            faces_dir = os.path.join(faces_folder, img_name + '_' + str(num_faces) + '.jpg')
            cv2.imwrite(faces_dir, cut)
            num_faces += 1
        num_img_with_faces += 1
        print rects

    print "Number of pictures with faces {}/{}".format(num_img_with_faces, len(images))


def unifyFaceFilenames(faces_folder=u'faces', base_width=62):
    """ unify the faces filenames and sizes after calling
        preprocess funcion and making sure each original image
        only maps to one preprocessed face image

    Parameters
    ----------
    faces_folder : string, optional, default u'faces'
        path to the folder containing the faces cropped from the original images
    base_width : int, optional, default 62
        target width for the resized square shaped face image
    """

    faces_filename = os.path.join(faces_folder, '*.jpg')
    images = glob.glob(faces_filename)
    images = map(lambda x: x[x.rfind("\\")+1:], images)
    for img_name in images:
        new_img_name = re.sub("_\d+", "", img_name)
        old_path = os.path.join(faces_folder, img_name)
        new_path = os.path.join(faces_folder, new_img_name)
        os.rename(old_path, new_path)
        img = Image.open(new_path)
        img = img.resize((base_width, base_width), Image.ANTIALIAS)
        img.save(new_path)
