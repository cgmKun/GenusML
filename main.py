import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib import cm
import json


with open("small-defects-query.json", "r") as read_file:
    response = json.load(read_file)
    response = json.loads(json.dumps(response))

data = response["data"]["defectsByReportId"]
defects = []
for val in data:
    defects.append({
        "id": val.get('_id'),
        "issueKey": val.get('issueKey'),
        "status": val.get('status'),
        "description": val.get('description')
    })

print(defects)
defects_df = pd.DataFrame(defects)
print(defects_df)

