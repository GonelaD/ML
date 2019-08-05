import numpy as np
import csv
import pandas as pd
import string 
import statistics

with open("linregdata") as f:
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]

row = ['gender','A2','A3','A4','A5','A6','A7','A8','A9']

data1 = []
data1.append(row)
for i in range(len(data)):
	data1.append(data[i])

with open("linregdata_duplicate", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(data1)

data = pd.read_csv("linregdata_duplicate")

Gender = {'M': (0,0,1),'F': (1,0,0),'I':(0,1,0)} 
data.gender = [Gender[item] for item in data.gender] 
print data.gender