#Standard Libraries Required
import pandas
import sys
import numpy as np
import pandas as pd
import requests
import json

# SKLEARN LIBRARIES
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

#GLOBAL VARIABLES
reportId = sys.argv[1]
true_k = sys.argv[2]

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

#Fetch Report
df = fetchReport("626c59cb164957a16a6e2d87")
print(type(reportId))

# STOP WORDS? + VECTORIZER
vectorizer = TfidfVectorizer(stop_words={'english', 'spanish'})
X = vectorizer.fit_transform(df.description)

# Sum_of_squared_distances = []
# K = range(2,10)
# for k in K:
#     km = KMeans(n_clusters=k, max_iter=200, n_init=10)
#     km = km.fit(X)
#     Sum_of_squared_distances.append(km.inertia_)

model = KMeans(n_clusters = int(true_k), init = 'k-means++', max_iter = 200, n_init = 10)
model.fit(X)
labels = model.labels_
clustered_defects = pd.DataFrame(list(zip(df._id, df.description, labels)),columns=['_id', 'description','group'])

for k in range(0,int(true_k)):
    print("cluster")
    mask = clustered_defects['group'] == k
    test = clustered_defects[mask]
