#Standard Libraries
import sys
import pandas as pd
import requests
import json
import re

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Keywords
from summa import keywords

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Global Variables - Command Line Arguments
reportId = sys.argv[1]
sessionId = sys.argv[2]
true_k = sys.argv[3]
session_date = sys.argv[4]

# Fetch defects by reportID GraphQL query
# Returns a dataframe with the query results
def fetchReport():
  query = """
    query {{
      defectsByReportId(reportId: "{reportId}") {{
        _id
        description
      }}
    }}    
  """.format(reportId = reportId)

  url = "http://localhost:8000/api"
  r = requests.post(url, json={'query': query})
  jsonData = json.loads(r.text)
  df_data = jsonData['data']['defectsByReportId']
  df = pd.DataFrame(df_data)
  return df

def submitGroup(group_df, group_key, keywords):
  groupTitle = "Group " + str(group_key+1)
  ids = group_df['_id']
  defect_ids = ""
  keywords_input = ""

  for i in range(len(ids)):
    defect_ids+=""""{defectId}", """.format(defectId = ids.iloc[i])

  for i in range(len(keywords)):
    keywords_input += """"{keyword}", """.format(keyword = keywords[i])
    if (i == 9):
      break

  query = """
    mutation {{
      createGroup(groupInput: {{
        groupTitle: "{groupTitle}"
        sessionId: "{sessionId}"
        submitDate: "{submitDate}"
        defects: [{defectIds}]
        keywords: [{keywords}]
        linkedReport: "{reportId}"
      }}) {{
        _id
      }}
    }}
  """.format(
    groupTitle = groupTitle,
    sessionId = sessionId,
    submitDate = session_date,
    defectIds = defect_ids,
    keywords = keywords_input,
    reportId = reportId
  )

  url = "http://localhost:8000/api"
  r = requests.post(url, json={'query': query})
  print(r)

def concatenateKeywords(keywords):
  keyword_list = []

  for i in range(0, len(keywords)):
    keyword_list.append(keywords[i][0] + ": " + str(keywords[i][1]))

  return keyword_list


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
  defects = fetchReport()

  #Vectorize the description from the defects dataframe
  vectorizer = TfidfVectorizer(stop_words={'english', 'spanish'})
  vectorized_defects_description = vectorizer.fit_transform(defects.description)

  # Setup and run K-Means Clustering Model - Unsupervised Model
  model = KMeans(n_clusters = int(true_k), init = 'k-means++', max_iter = 200, n_init = 10)
  model.fit(vectorized_defects_description)
  labels = model.labels_

  # Retrieve the clustered defects in a dataframe
  clustered_defects = pd.DataFrame(list(zip(defects._id, defects.description, labels)),columns=['_id', 'description','group'])

  # Most Popular Labels
  result={'cluster': labels, 'defect_description':defects.description}
  result=pd.DataFrame(result)

  # Split each cluster into individual dataframes
  for k in range(0,int(true_k)):
    # Filtered Cluster
    new_df = clustered_defects.loc[(clustered_defects['group'] == k)]

    # Keywords
    s=result[result.cluster==k]
    raw_text=s['defect_description'].str.cat(sep=' ')
    raw_text=raw_text.lower()
    raw_text=' '.join([word for word in raw_text.split()])
    TR_keywords = keywords.keywords(raw_text, scores=True)
    submitGroup(new_df, k, concatenateKeywords(TR_keywords))

main()

#https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
#https://towardsdatascience.com/clustering-documents-with-python-97314ad6a78d
#https://github.com/dpanagop/ML_and_AI_examples/blob/master/NLP_example_clustering.ipynb
