
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
class knn:
    def kn(self):
        print("---KNN Classification Algorithm---")
        # Import the data
        dataset = pd.read_csv('Social_Network_Ads.csv')
        x = dataset.iloc[:, [2,3]].values
        y = dataset.iloc[:, 4].values
        # train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
        # Feature Scaling
        scalar = StandardScaler()
        x_train = scalar.fit_transform(x_train)
        x_test = scalar.transform(x_test)
        # Perform KNN
        knn = KNeighborsClassifier(n_neighbors = 5, p=2,metric='minkowski')
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("confusion_matrix")
        print(cm)
        # Calculating the Accuracy
        print("accuracy")
        print(accuracy_score(y_test, y_pred))
        return  accuracy_score(y_test, y_pred)
    def info(self):
        dataset = pd.read_csv('Social_Network_Ads.csv')
        #print(dataset.describe())
        print(dataset.head(10))