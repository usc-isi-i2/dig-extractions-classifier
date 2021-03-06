import os

import digExtractionsClassifier.utility.functions as utility_functions
from digExtractionsClassifier import dig_extractions_classifier
from sklearn.externals import joblib


class ProcessClassifier():
    """ Class to process the classifiers """
    def __init__(self, extraction_classifiers):
        self.extraction_classifiers = extraction_classifiers
        self.embeddings_file = 'unigram-part-00000-v2.json'
        self.__initialize()

    def __initialize(self):
        """ Initialize classifiers """
        self.embeddings = utility_functions.load_embeddings(self.embeddings_file)
        self.extractors = []

        for classifier in self.extraction_classifiers:
            print "Setting up - " + classifier
            extractor = self.setup_classifier(classifier)
            self.extractors.append(extractor)

    def classify_extractions(self, doc):

        if 'knowledge_graph' in doc:
            knowledge_graph = doc['knowledge_graph']

            for extractor in self.extractors:
                extractor.classify(knowledge_graph)

        return doc

    def setup_classifier(self, classification_field):
        directory = os.path.dirname(os.path.abspath(__file__))
        MODEL_FILE = classification_field + '.pkl'
        model_file_path = os.path.join(directory, 'resources', MODEL_FILE)

        model = self.load_model([model_file_path])

        return dig_extractions_classifier.DigExtractionsClassifier(model, classification_field, self.embeddings)

    def load_model(self, filenames):
        print filenames
        classifier = dict()
        # try:
        filename = utility_functions.get_full_filename(__file__, filenames[0])
        print filename
        classifier['model'] = joblib.load(filename)
        # except:
        # raise Exception('Model file not present')
        try:
            classifier['scaler'] = joblib.load(utility_functions.get_full_filename(__file__, filenames[1]))
            classifier['normalizer'] = joblib.load(utility_functions.get_full_filename(__file__, filenames[2]))
            classifier['k_best'] = joblib.load(utility_functions.get_full_filename(__file__, filenames[3]))
        except:
            pass
        return classifier
