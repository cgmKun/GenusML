import pandas
import tensorflow as tf
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
    _id
    issueKey
    status
    priority
    severity
    projectKey
    issueType
    created
    assignee
    digitalService
    summary
    description
  }
}"""

url = "http://localhost:8000/api"
r = requests.post(url, json={'query': query})
jsonData = json.loads(r.text)
df_data = jsonData['data']['defectsByReportId']
df = pandas.DataFrame(df_data)
print(df)


# Tokenization
tokenizedDF = []
words = re.findall(r'\w+', df)

for word in words:
  if not word in tokens:
    tokens[word] = len(tokens) + 1
  
  tokenizedDF.append(tokens[word])

# print tokenized df
for i in tokenizedDF:
  print(i)
  