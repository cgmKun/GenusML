import pandas
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib import cm
import json
import re

# Map of tokens
tokens = {}

query = """query	{
  defectsByReportId(reportId: "6273626676b6958fa71d9315") {
    description
  }
}""" 


url = "http://localhost:8000/api"
r = requests.post(url, json={'query': query})
jsonData = json.loads(r.text)
df_data = jsonData['data']['defectsByReportId']
print(df_data)
df = pandas.DataFrame(df_data)
print(df)


#_id
    #issueKey
    #status
    #priority
    #severity
    #projectKey
    #issueType
    #created
    #assignee
    #digitalService
    #summary

# Tokenization
tokenizedDF = []
lowerDF = df.lower()
words = re.findall(r'\w+', lowerDF)

for word in words:
  if not word in tokens:
    tokens[word] = len(tokens) + 1
  
  tokenizedDF.append(tokens[word])

# print tokenized df
for i in tokenizedDF:
  print(i)

