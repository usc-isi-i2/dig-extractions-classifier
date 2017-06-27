import codecs
import json

from initClassifiers import ProcessClassifier

def load_json_file(file_name):
    rules = json.load(codecs.open(file_name, 'r', 'utf-8'))
    return rules

embeddings_file = 'word_embeddings_cr.wordembed'
extraction_classifiers = ['city', 'ethnicity', 'hair_color', 'name', 'eye_color']

classifier_processor = ProcessClassifier(extraction_classifiers, embeddings_file)

# properties = load_json_file('resources/properties_non_cluster.json')

input_file = '/home/vinay/Documents/ISI/etk/etk/classifiers/processed_australia.jl'
output_file = '/home/vinay/Documents/ISI/etk/etk/classifiers/processed_australia_classified.jl'

output = codecs.open(output_file, 'w', 'utf-8')
with codecs.open(input_file, 'r', 'utf-8') as f:
    for index, line in enumerate(f):
        result_doc = json.loads(line)

        result_doc = classifier_processor.classify_extractions(result_doc)

        print index

        json.dump(result_doc, output)
        output.write('\n')

output.close()
