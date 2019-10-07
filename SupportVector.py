import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
class supportvector:
    def svc(self):
        print("---SupportVector Classification Algorithm---")
        dataset = pd.read_csv('Social_Network_Ads.csv')
        x = dataset.iloc[:, [2,3]]
        y = dataset.iloc[:, 4]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("accuracy")
        print(accuracy_score(y_pred, y_test))
        return accuracy_score(y_test, y_pred)
