import tensorflow as tf
import utils.detect_face
import cv2
import numpy as np
from utils.utils import prewhiten

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

class Cropper():

    def __init__(self):
        self.cropped_ref_images=None
        self.cropped_test_images=None

        self.image_size = 160
        self.margin = 44
        self.gpu_memory_fraction =1.0

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor


    def cropp_image(self, dict_imgs):

        #np_load_old = np.load
        #np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        dict_cropped ={}

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction= self.gpu_memory_fraction)
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = utils.detect_face.create_mtcnn(sess, None)

        for key, img in dict_imgs.items():

            img_list = []
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = utils.detect_face.detect_face(img, self.minsize, pnet, rnet, onet, self.threshold, self.factor)
            if len(bounding_boxes) < 1:
                print("can't detect face, remove ", image)

            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            margin = self.margin
            image_size= self.image_size

            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = cv2.resize(cropped, (image_size, image_size))
            prewhitened = prewhiten(aligned)
            img_list.append(prewhitened)
            image = np.stack(img_list)
            dict_cropped[key]=image

        return dict_cropped

    def cropp_ref_images(self, dict_ref_images):
        self.cropped_ref_images = self.cropp_image(dict_ref_images)

    def cropp_test_images(self, dict_test_images):
        self.cropped_test_images = self.cropp_image(dict_test_images)

