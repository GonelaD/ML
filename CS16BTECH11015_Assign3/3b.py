import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import math 
import csv
import random
from sklearn.linear_model import Ridge
from statistics import mean,stdev


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

# ------------------------3b----------------------#

Gender = {'M': 2,'F': 1,'I':0} 
data.gender = [Gender[item] for item in data.gender] 

mean=(data.mean(axis=0))
SD = (data.std(axis=0))
data.gender = (data.gender-mean.gender)/SD.gender
data.A2 = (data.A2-mean.A2)/SD.A2
data.A3 = (data.A3-mean.A3)/SD.A3
data.A4 = (data.A4-mean.A4)/SD.A4
data.A5 = (data.A5-mean.A5)/SD.A5
data.A6 = (data.A6-mean.A6)/SD.A6
data.A7 = (data.A7-mean.A7)/SD.A7
data.A8 = (data.A8-mean.A8)/SD.A8

print data

