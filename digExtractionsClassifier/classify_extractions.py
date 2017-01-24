from operator import add
import numpy as np

import digExtractionsClassifier.utility.functions as utility_functions

class ClassifyExtractions:
    def __init__(self, model, classification_field, embeddings, context_range = 5, use_word_in_context = True, use_break_tokens = False):
        self.model = model
        self.classification_field = classification_field
        self.embeddings = embeddings
        self.context_range = context_range
        self.use_word_in_context = use_word_in_context
        self.use_break_tokens = use_break_tokens

    def classify(self, tokens):
        # tokens = map(lambda x:x.lower(),tokens)
        for index, token in enumerate(tokens):
            utility_functions.value_to_lower(token)
            semantic_types = utility_functions.get_extractions_of_type(token, self.classification_field)
            for semantic_type in semantic_types:
                #There are extractions in the token of the same type
                length = utility_functions.get_length_of_extraction(semantic_type)
                context = utility_functions.get_context(tokens, index, length, self.context_range, self.use_word_in_context)
                context_vector = utility_functions.get_vector_of_context(context, self.embeddings)
                probability = self.get_classification_probability(context_vector)
                self.append_probability(semantic_type, probability)
        
        return tokens

    def get_classification_probability(self, feature_vectors):
        if(feature_vectors.ndim == 1):
            # print "Reshaping the vector"
            feature_vectors = feature_vectors.reshape(1, -1)

        if('scaler' in self.model):
            print "Scaling"
            feature_vectors = self.model['scaler'].transform(feature_vectors)
        
        if('normalizer' in self.model):
            print "Normalizing"
            feature_vectors = self.model['normalizer'].transform(feature_vectors)

        if('k_best' in self.model):
            feature_vectors = self.model['k_best'].transform(feature_vectors)
    
        return self.model['model'].predict_proba(feature_vectors)[0][1]

    def append_probability(self, semantic_type, probability):
        semantic_type['probability'] = probability

   