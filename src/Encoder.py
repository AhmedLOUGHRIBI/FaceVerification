from consts import Consts
import tensorflow as tf


class Encoder():

    def __init__(self, loader):
        self.encoding_ref_images = None
        self.encoding_test_images = None
        self.loader = loader #must have loaded model


    def calculate_encodings(self, dict_images_cropped):
    
        model = Consts.model_path
        dict_enc={}
        for key,image_cropped in dict_images_cropped.items():

            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    # Load the model
                    self.loader.load_model(model)
                    # Get input and output tensors
                    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: image_cropped, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

            dict_enc[key]=emb
        return dict_enc


    def calculate_ref_encodings(self,dict_ref_images):
        self.encoding_ref_images = self.calculate_encodings(dict_ref_images)


    def calculate_test_encodings(self,dict_test_images):
        self.encoding_test_images = self.calculate_encodings(dict_test_images)
