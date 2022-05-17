import pandas
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib import cm
import json

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