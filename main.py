#Standard Libraries
import pandas
import sys
import numpy as np
import pandas as pd
import requests
import json

# Sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Global Variables - Command Line Arguments
reportId = sys.argv[1]
true_k = sys.argv[2]

# Fetch defects by reportID GraphQL query
# Returns a dataframe with the query results
def fetchReport(reportId):
    query = """query    {
        defectsByReportId(reportId: "%s") {
            _id
            description
        }
    }"""

    url = "http://localhost:8000/api"
    r = requests.post(url, json={'query': query % (reportId)})
    jsonData = json.loads(r.text)
    df_data = jsonData['data']['defectsByReportId']
    df = pd.DataFrame(df_data)
    return df

# Map of tokens is global to maintain uniqueness of tokens across multiple calls to method
tokens = {}

# Receives a string and returns an array of tokens for each word on string
def tokenize(str):
  # Output array
  tokenizedDF = []

  # Lowercases all chars
  lowerDF = str.lower()

  # Regex hanldes special characters
  words = re.findall(r'\w+', lowerDF)

  # Tokenizes and appends to output array
  for word in words:
    if not word in tokens:
      tokens[word] = len(tokens) + 1
  
    tokenizedDF.append(tokens[word])

  # Print tokenized string
  for i in tokenizedDF:
    print(i)

def main():
    #Fetch Report data
    defects = fetchReport(reportId)

    #Vectorize the description from the defects dataframe
    vectorizer = TfidfVectorizer(stop_words={'english', 'spanish'})
    vectorized_defects_description = vectorizer.fit_transform(defects.description)

    # Setup and run K-Means Clustering Model - Unsupervised Model
    model = KMeans(n_clusters = int(true_k), init = 'k-means++', max_iter = 200, n_init = 10)
    model.fit(vectorized_defects_description)
    labels = model.labels_

    # Retrieve the clustered defects in a dataframe
    clustered_defects = pd.DataFrame(list(zip(defects._id, defects.description, labels)),columns=['_id', 'description','group'])

    # Split each cluster into individual dataframes
    for k in range(0,int(true_k)):
        mask = clustered_defects['group'] == k
        test = clustered_defects[mask]
        print(test)

main()
