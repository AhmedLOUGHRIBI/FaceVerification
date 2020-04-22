from inout.input import DataLoader
import Cropper
import Encoder
import Classifier
from inout.output import DataWriter


loader = DataLoader.DataLoader()
loader.load_all_images()

cropper = Cropper.Cropper()
cropper.cropp_ref_images(loader.ref_images)
cropper.cropp_test_images(loader.test_images)

reference_cropped = cropper.cropped_ref_images
test_cropped = cropper.cropped_test_images


encoder = Encoder.Encoder(loader)
encoder.calculate_ref_encodings(reference_cropped)
encoder.calculate_test_encodings(test_cropped)


ref_encoded = encoder.encoding_ref_images
test_encoded = encoder.encoding_test_images


classifier = Classifier.Classifier()
classifier.Classify(ref_encoded, test_encoded)

writer = DataWriter.DataWriter()
writer.write_as_csv(classifier.test_classes,'test_classes', 'CLASSE')