from consts import Consts
from os import listdir
from os.path import isfile, join, expanduser
from utils.utils import LoadImage, get_model_filenames
import tensorflow as tf



class DataLoader():

    def __init__(self):
        self.ref_images = None
        self.test_images = None

    def load_all_images(self):
        self.load_reference_images()
        self.load_test_images()

    def load_reference_images(self):
        image_folder=Consts.reference_images_folder
        onlyfiles = [image_folder + '/' + str(f) for f in listdir(image_folder) if isfile(join(image_folder, f))]
        self.ref_images = {}
        for image_path in onlyfiles:
            image = LoadImage(image_path)
            key = image_path.split('{0}/'.format(image_folder))[1]
            self.ref_images[key.split('.')[0]] = image


    def load_test_images(self):

        image_folder = Consts.retraits_images_folder
        onlyfiles = [image_folder + '/' + str(f) for f in listdir(image_folder) if isfile(join(image_folder, f))]
        self.test_images = {}
        for image_path in onlyfiles:
            image = LoadImage(image_path)
            key = image_path.split('{0}/'.format(image_folder))[1]
            self.test_images[key.split('.')[0]] = image


    def load_model(self, model, input_map=None):
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = expanduser(model)
        if (isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=input_map, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.compat.v1.train.import_meta_graph(join(model_exp, meta_file), input_map=input_map)
            saver.restore(tf.compat.v1.get_default_session(), join(model_exp, ckpt_file))