# dig-extractions-classifier

Uses the context of the extractions to give a probability using word embeddings

## Running the Extractions Classifier

* Useful Imports

```
from digExtractionsClassifier import dig_extractions_classifier
import digExtractionsClassifier.utility.functions as utility_functions
from sklearn.externals import joblib
```

* Useful Functions

```
  def setup_classifier(self, classification_field):
  """ 
  Setup the classifier for the given field
  """
    directory = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILE = classification_field+'.pkl'
    model_file_path = os.path.join(directory, 'resources', MODEL_FILE)

    model = self.load_model([model_file_path]) #Passing just the model filepath for our use case

    return dig_extractions_classifier.DigExtractionsClassifier(model, classification_field, self.embeddings)

  def load_model(self, filenames):
  """ 
  Loads the model files
  :param filenames: Assumes a list of file names of model file, [scaler], [normalizer], [k_best] (Last 3 optional)
  :return classifier: Classifier Object
  """
    classifier = dict()
    filename = utility_functions.get_full_filename(__file__, filenames[0])
    classifier['model'] = joblib.load(filename)
    
    try:
      classifier['scaler'] = joblib.load(utility_functions.get_full_filename(__file__, filenames[1]))
      classifier['normalizer'] = joblib.load(utility_functions.get_full_filename(__file__, filenames[2]))
      classifier['k_best'] = joblib.load(utility_functions.get_full_filename(__file__, filenames[3]))
    except:
      pass
    return classifier
```

* Define the Classifier Types and Embeddings File

.pkl files for each classifier must be present

```
self.extraction_classifiers = ['city', 'ethnicity', 'hair_color', 'name', 'eye_color']
self.embeddings_file = 'unigram-part-00000-v2.json'
```

* Load Embeddings File

```
self.embeddings = utility_functions.load_embeddings(self.embeddings_file)
```

* Setup Classifiers

```
for classifier in self.extraction_classifiers:
  print "Setting up - "+classifier
  extractor = setup_classifier(classifier)
  extractors.append(extractor)
```

* Call the classifiers over knowledge_graphs

```
for extractor in self.extractors:
  extractor.classify(knowledge_graph)
        
```

