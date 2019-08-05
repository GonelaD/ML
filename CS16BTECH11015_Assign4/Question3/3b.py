import numpy
import matplotlib.pyplot as plt 

def MyDBSCAN(D, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue
        
        # Find all of P's neighboring points.
        NeighborPts = regionQuery(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else: 
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    
    return labels


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    
    labels[P] = C
    
    i = 0
    while i < len(NeighborPts):    
        
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        
        elif labels[Pn] == 0:
            labels[Pn] = C
            
            PnNeighborPts = regionQuery(D, Pn, eps)
            
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            
        i += 1        
    

def regionQuery(D, P, eps):
    
    neighbors = []
    for Pn in range(0, len(D)):
        
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors

if __name__ == "__main__":
    train_data = numpy.genfromtxt('dataset1.txt')
    y=MyDBSCAN(train_data,0.1,10)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=y, s=50, cmap='viridis')
    plt.savefig('DBSCAN for dataset1.png')
    plt.show()

    train_data = numpy.genfromtxt('dataset2.txt')
    y=MyDBSCAN(train_data,1,10)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=y, s=50, cmap='viridis')
    plt.savefig('DBSCAN for dataset2.png')
    plt.show()


