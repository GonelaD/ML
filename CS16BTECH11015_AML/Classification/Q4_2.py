import numpy as np
import pandas as pd
import json
from sklearn.naive_bayes import *
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


with open('train.json') as data_file:
	train_data = json.load(data_file)

with open('test.json') as data_file1:
	test_data = json.load(data_file1)

def countmatrix(dict,num_cuisines,num_ingredients,cuisines,ingredients,train_data):

	count_matrix = np.zeros((num_cuisines,num_ingredients))

	row=0

	for cuisine in cuisines:
		ingredients_per_cuisine = dict[cuisine]

		for ingred in ingredients_per_cuisine:
			
			column=ingredients.index(ingred)
			count_matrix[row,column] = count_matrix[row,column]+1

		row += 1

	return count_matrix




#step1: Create ingredients for each cuisine
def ingred_for_cuisine(data):
	cuisine_and_ingred={}
	#all cuisines
	cuisines=[]
	# all ingredients
	ingredients=[]

	for i in range(len(data)):
		cuisine = data[i]['cuisine']
		ingred_per_cuisine = data[i]['ingredients']

		if cuisine not in cuisine_and_ingred.keys():
			cuisine_and_ingred[cuisine] = ingred_per_cuisine
			cuisines.append(cuisine)
		else:
			cuisine_and_ingred[cuisine].extend(ingred_per_cuisine)

		ingredients.extend(ingred_per_cuisine)

	#unique ingredients
	ingredients = list(set(ingredients))
	num_cuisines = len(cuisines)
	num_ingredients = len(ingredients)

	return cuisine_and_ingred,num_cuisines,num_ingredients,cuisines,ingredients





if __name__ == "__main__":
	X_dictCuisineIngred , num_cuisines,num_ingredients,cuisines,ingredients =ingred_for_cuisine(train_data)
	X_countmatrix = countmatrix(X_dictCuisineIngred,num_cuisines,num_ingredients,cuisines,ingredients,train_data)
	#print X_countmatrix[0]
	#fopen("result.csv","w")
	
	
	X_test =[[0 for x in range(num_ingredients)] for y in range(len(test_data))]

	row1=0
	for i in range(len(test_data)):
		for row in test_data[i]['ingredients']:
			if row in ingredients:
				column1=ingredients.index(row)
				X_test[row1][column1] += 1
		row1 += 1

	clf = MultinomialNB()
	#print type(X_countmatrix)
	#print type(cuisines)
	clf.fit(X_countmatrix,cuisines)
	y_test = clf.predict(X_test)
	f = open('results.csv','w')
	f.write(u'id,cuisine\n')
	for i in range(len(test_data)):
		f.write('%s,%s\n'% (test_data[i]['id'],y_test[i]))
	f.close()
