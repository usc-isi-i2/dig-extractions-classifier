import codecs
import json

from initClassifiers import ProcessClassifier

def load_json_file(file_name):
    rules = json.load(codecs.open(file_name, 'r', 'utf-8'))
    return rules

extraction_classifiers = ['city', 'ethnicity', 'hair_color', 'name', 'eye_color']

classifier_processor = ProcessClassifier(extraction_classifiers)

properties = load_json_file('properties_non_cluster.json')

output = codecs.open('output_ground_truth.jl', 'w', 'utf-8')
with codecs.open('ground_truth_extractions.jl', 'r', 'utf-8') as f:
    for index, line in enumerate(f):
        result_doc = json.loads(line)

        result_doc = classifier_processor.classify_extractions(result_doc)

        print index

        json.dump(result_doc, output)
        output.write('\n')

output.close()
