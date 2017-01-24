try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'digExtractionClassifier',
    'description': 'digExtractionClassifier',
    'author': 'Rahul Kapoor',
    'url': 'https://github.com/usc-isi-i2/dig-extractions-classifier',
    'download_url': 'https://github.com/usc-isi-i2/dig-extractions-classifier',
    'author_email': 'rahulkap@isi.edu',
    'version': '0.3.0',
    'install_requires': ['digExtractor>=0.3.0'],
    # these are the subdirs of the current directory that we care about
    'packages': ['digExtractionsClassifier'],
    'scripts': [],
}

setup(**config)
