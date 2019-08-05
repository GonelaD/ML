# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
import math 
from random import seed
from random import randrange


# Enter You Name Here
myname = "CS16BTECH11015" # or "Doe-Jane-"

# Implement your decision tree below

# class find_ques:
#     """docstring for find_ques"""
#     def __init__(self, column_no,value):
#         self.column_no  = column_no
#         self.value =  value

#     def match(self,example):








# def partition(rows, ques):
#     true,false = [],[]

#     for row in rows:
#         if ques.match(row):
#             true.append(row)
#         else:
#             false.append(row)
    
#     return true,false


#splits based on attribute and value 
def test_split(col_no,value,rows):

    l_set,r_set = list(),list()
    
    for row in rows:
        #val:dividing based on ; col_no:which particular attribute is checked
        if row[col_no] < value:
            l_set.append(row)
        else:
            r_set.append(row)              

    return l_set,r_set

def terminal(group):
    
    outcome = [row[-1] for row in group]

    return max(set(outcome),key=outcome.count)

def gini(groups,results):

    total_size = float(sum([len(group) for group in groups]))

    gini = 0.0

    # one particular group
    for group in groups:   
         #for denominator
        size = float(len(group))

        if size==0:
            continue

        score = 0.0
        

        # last column of the actual database
        for result in results:
            p=0.0
            #traversing through particular group
            for row in group:
                if row[-1]==result:
                    p=p+1.0
            p = p/size
            # for each group
            if p!=0:
                score = score + p*p
            else:
                score = score + 0.0

        score = 1.0 - score

        gini += score * (size/total_size)

    return gini


#select the best one
def best_split(rows):

    results = list(set(row[-1] for  row in rows))  #last column

    n_col = len(rows[0]) - 1 #number of columns
    #print n_col
    attr,value,e_val,info_gain,g_score,final_group =  100,1000,1,0,0,None

    for col in range(n_col):

        unique_values = list(set([row[col] for row in rows])) #set gives unique values
        #print unique_values

        for val in unique_values:

            groups = test_split(col,val,rows)

            # do entropy
            # check = check_entropy(groups,results)

            # if e_val-check > info_gain:
            #     info_gain = e_val-check
            #     attr,value,final_group  = col,val,groups

            check = gini(groups,results)
            if check > g_score:
                g_score=check
                attr,value,final_group = col,val,groups


   # print "\n"

    # return attribute(column number), value(to be divided on), group(best fit group with left and right - node)
    #print attr,value
    return {'index':attr,'value':value,'group':final_group}


def split(node,max_depth,min_size,depth):
    left,right = node['group']
    del(node['group'])
    # print left
    # print right
    if not left or not right:
        node['left']=node['right'] = terminal(left+right)
        return

    if depth >= max_depth:
        node['left'],node['right']=terminal(left),terminal(right)
        return

    if len(left)<=min_size:
        node['left']=terminal(left)
       # print "left_terminal"
    else:
        node['left']=best_split(left)
        split(node['left'],max_depth,min_size,depth+1)

    if len(right)<=min_size:
       # print "right_terminal"
        node['right']=terminal(right)
    else:
        node['right']=best_split(right)
        split(node['right'],max_depth,min_size,depth+1)



#to build tree
def build_tree(training_set,max_depth,min_size):
    #get the best split
    root = best_split(training_set)
    #print root
    #spliting it
    split(root,max_depth,min_size,1)
    
    #returning the root node
    return root

def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


def predict(node,test_instance):

    if test_instance[node['index']]< node['value']:
            if isinstance(node['left'],dict):
                return predict(node['left'],test_instance)
            else:
                #on reaching a terminal
                return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],test_instance)
        else:
            return node['right']

class DecisionTree():
    tree = {}

    def learn(self, training_set):
        # implement this function
        self.tree = build_tree(training_set,max_depth,min_size)
        #print_tree(self.tree)
       


    # implement this function
    def classify(self, test_instance):

        result = predict(self.tree,test_instance)
        #print result
        return result

        

def run_decision_tree(max_depth,min_size,n_folds):

    #Load dataset
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print "Number of records: %d" % len(data)


    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    accuracy = 0.0
    training_set=[x for i,x in enumerate(data) if (i)%K != 9]
    test_set = [x for i,x in enumerate(data) if (i)%K == 9]

    # for j in range(1,11):
    #     training_set=[x for i,x in enumerate(data) if (i+j)%K != 9]
    #     test_set = [x for i,x in enumerate(data) if (i+j)%K == 9]
    tree = DecisionTree()
        # Construct a tree using training set
    tree.learn( training_set )
            # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])
        # Accuracy
    accuracy = float(results.count(True))/float(len(results))
        #print "accuracy: %.4f" % accuracy    
    #accuracy = accuracy/10.0
    print accuracy
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result3c.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    n_folds=1
    max_depth=20
    min_size=1
    run_decision_tree(max_depth,min_size,n_folds)
 