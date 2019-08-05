import csv
import numpy as np
import math 
from random import seed
from random import randrange
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


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

def check_entropy(groups,results):

    total_size = float(sum([len(group) for group in groups]))

    total_entropy = 0.0

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
                score = score + p*(math.log(p,2))
            else:
                score = score + 0.0

        score *= -1

        total_entropy += score * (size/total_size)

    return total_entropy

def terminal(group):
    
    outcome = [row[-1] for row in group]

    return max(set(outcome),key=outcome.count)


#select the best one
def best_split(rows,n_features):

    results = list(set(row[-1] for  row in rows))  #last column

    #print n_col
    attr,value,e_val,info_gain,g_score,f_left,f_right =  100,1000,1,0,10,None,None

    for col in range(n_features):

        unique_values = list(set([row[col] for row in rows])) #set gives unique values
        #print unique_values

        for val in unique_values:

            groups = test_split(col,val,rows)
            left,right = groups

            # do entropy
            check = check_entropy(groups,results)

            if e_val-check >= info_gain:
                info_gain = e_val-check
                attr  = col
                value = val
                f_left = left
                f_right = right

            # check = gini(groups,results)
            # if check < g_score:
            #     attr,value,final_group = col,val,groups


   # print "\n"

    # return attribute(column number), value(to be divided on), group(best fit group with left and right - node)
    return {'index':attr,'value':value,'left':f_left,'right':f_right}


def split(node,min_size,depth,n_features):
    left = node['left']
    right = node['right']
    
    if not left or not right:
        
        if not left and not right:
            return
        else:
            node['left']=node['right'] = terminal(left+right)
            return

    else:
        node['left'] = best_split(left,n_features)
        split(node['left'],min_size,depth+1,n_features)
        node['right'] = best_split(right,n_features)
        split(node['right'],min_size,depth+1,n_features)



#to build tree
def build_tree(training_set,min_size,n_features):
    #get the best split
    root = best_split(training_set,n_features)
    #print root
    #spliting it
    split(root,min_size,1,n_features)
    
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

    def learn(self, training_set,n_features):
        # implement this function
        self.tree = build_tree(training_set,min_size,n_features)
        return self.tree

        #print_tree(self.tree)
       


    # implement this function
    


        



def bagging_and_predict(trees,row):
    predictions = [predict(tree,row) for tree in trees]
    return max(set(predictions),key=predictions.count)

def subsample(train,ratio):
    sample = []
    n_sample = round(len(train) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(train)-1)
       # print index
        sample.append(train[index])
    return sample


def random_forests(data,min_size,n_features):
    K=10
    training = [data[i] for i in range(len(data)) if (i)%K != 0 and (i)%K !=1 and (i)%K !=2]
    testing = [data[i] for i in range(len(data)) if (i)%K == 0 or (i)%K ==1 or (i)%K ==2]
    trees = []
    tree_number = 0
    sample = []
  
    for i in range(5):
        sample.append(subsample(training,0.3))
        tree = DecisionTree()
        #print_tree(sample)
        trees.append(tree.learn(sample[tree_number],n_features))
        tree_number += 1


    predictions = [bagging_and_predict(trees,row[:-1]) for row in testing]
    results=[]
    for i in range(len(testing)):
        if testing[i][len(testing[0])-1]==predictions[i]:
            results.append(True)
        else:
            results.append(False)


    accuracy = float(results.count(True))/float(len(results))
        #print "accuracy: %.4f" % accuracy    
    #accuracy = accuracy/10.0
    print "Accuracy:",accuracy


if __name__ == "__main__":
    
    min_size=1
    data = np.genfromtxt("data")
    n_features = [20,25,30,35,40,45,50,57]
    for i in range(len(n_features)):
        print "m:",n_features[i]
        random_forests(data,min_size,n_features[i])

