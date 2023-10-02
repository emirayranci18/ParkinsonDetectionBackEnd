import pandas as pd

data = pd.read_csv("cleanedParkinson.data")

data = data.drop(['name', 'status'], axis = 1)

data.to_csv("test2.txt", header=False)
