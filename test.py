#Standard Libraries
import json
from pathlib import Path

# Sklearn
from sklearn.feature_extraction import text


# Stopwords
script_location = Path(__file__).absolute().parent
spanish_location = script_location / 'stop_words_spanish.json'
english_location = script_location / 'stop_words_english.json'

spanish_json = spanish_location.open()
english_json = english_location.open()

spanish_stopwords = json.load(spanish_json)
english_stopwords = json.load(english_json)
complete_stopwords = spanish_stopwords + english_stopwords
my_stop_words = text.ENGLISH_STOP_WORDS.union(spanish_stopwords)

spanish_json.close()
english_json.close()

print("COMPLETE \n")
print(complete_stopwords)
print("SKLEAERN \n")
print(my_stop_words)