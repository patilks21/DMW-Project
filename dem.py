from KNN import knn
from RandomForest import randomforest
from SupportVector import supportvector
knn=knn()
b=knn.info()
a=knn.kn()
rf=randomforest()
b=rf.rf()
sv=supportvector()
c=sv.svc()
print("Accuracy of KNN is : "+str(a))
print("Accuracy of RandomForest is : "+str(b))
print("Accuracy of SupportVector is : "+str(c))


