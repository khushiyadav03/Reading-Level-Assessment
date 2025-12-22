import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('wordnet', quiet=False)
    nltk.download('omw-1.4', quiet=False) # valid for some nltk versions
    from nltk.corpus import wordnet
    print("WordNet downloaded successfully.")
except Exception as e:
    print(f"Error downloading WordNet: {e}")

