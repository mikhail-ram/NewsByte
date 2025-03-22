import logging
import nltk
import spacy
import os

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
# Add this directory to nltk's data path
nltk.data.path.append(nltk_data_dir)
# Download the required model to the new directory
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nlp_spacy = spacy.load("en_core_web_sm")

logger = logging.getLogger("NewsByte")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

DEBUG_MODE = True

if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
