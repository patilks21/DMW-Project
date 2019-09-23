import pandas as pd
data_set = pd.read_csv('/home/administrator/Downloads/diabetes.csv')
x = data_set.iloc[:,:-1]
y = data_set.iloc[:,-1]
x.head()
y.head()
x.info()
x.isnull().any()
#SCALING
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
#SPLITTING
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.25,random_state=0)
#CLASSIFIER (TRAINING)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)
GaussianNB(priors=None)
#PREDICTIONS (TESTING)
y_pred = classifier.predict(X_test)
#Validating Prediction using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
accuracy*100
from sklearn.metrics import precision_recall_fscore_support
prf = precision_recall_fscore_support(Y_test,y_pred)
print('Precision ',prf[0]*100)
Precision  [ 79.72027972  67.34693878]
print('Recall ',prf[1]*100)
Recall  [ 87.69230769  53.22580645]
print('F1 Measure', prf[2]*100)
F1 Measure [ 83.51648352  59.45945946]
print('Support', prf[3]*100)
Support [13000  6200]

