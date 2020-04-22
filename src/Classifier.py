from utils.utils import CalculateDistance


class Classifier:

    def __init__(self):
        self.test_classes = {}


    def Classify(self, dict_reference_encodings, dict_test_encodings):

        for key, encoding in dict_test_encodings.items():
            dist = CalculateDistance(encoding, dict_reference_encodings['Ahmed'])
            if dist < 1.1:
                self.test_classes[key] = 'This is Ahmed'
            else:
                self.test_classes[key] = 'This is not Ahmed'