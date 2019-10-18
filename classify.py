from KNN import knn
from RandomForest import randomforest
from SupportVector import supportvector
from matplotlib import pyplot as py
knn=knn()

a=knn.kn()
rf=randomforest()
b=rf.rf()
sv=supportvector()
c=sv.svc()
x=["KNN","RandomForest","SupportVector"]
y=[a*100,b*100,c*100]
py.bar(x,y)
py.title("Accuracy Comparision")
py.xlabel("Algorithm")
py.ylabel("Accuracy %")
py.show()
print("Accuracy of KNN is : "+str(a))
print("Accuracy of RandomForest is : "+str(b))
print("Accuracy of SupportVector is : "+str(c))


