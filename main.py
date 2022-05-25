import pandas
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib import cm
import json
import re

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
