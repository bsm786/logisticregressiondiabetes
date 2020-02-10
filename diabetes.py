import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#col_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
pima = pd.read_csv("diabetes.csv")
print (pima.head())
print (pima.isnull().values.any())


def plot_corr(df, size =10):
    corr= df.corr()
    fig, axis = plt.subplots(figsize=(size,size))
    axis.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns) #draw xticks
    plt.yticks(range(len(corr.columns)),corr.columns) #draw yticks
    
    
print (plot_corr(pima))
print (pima.corr())
    
sns.pairplot(pima,x_vars=['Pregnancies', 'Insulin','BMI','Age','Glucose','DiabetesPedigreeFunction','Age'],y_vars='Outcome',height =6, aspect = 0.7, kind='reg' )
    
#split dataset into features and target variable
col_features = ['Pregnancies', 'Insulin','BMI','Age','Glucose','DiabetesPedigreeFunction']
x = pima[col_features] #features
y = pima.Outcome #target

#split x and y into training and testing datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.25, random_state = 0)
#print (x_train)
#print (y_train)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000).fit(x_train,y_train)
y_pred = logreg.predict(x_test)

#confusion metrics helps to evaluate the accuracy of the classification model
from sklearn import metrics
cnf = metrics.confusion_matrix(y_test,y_pred)
print (cnf)

print ("Classification Report")
print ("Accuracy means Classification Accuracy rate of the model, in this case the Logistic Regression model has an accuracy of 79%.")
print ("Precision means how accurate your model is, in this case, the model predicts that 76% of the time a suspected patient will get diabetes.")
print ("Recall means  If there are patients who have diabetes in the test set and your Logistic Regression model can identify it 54% of the time")
print ("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test,y_pred)))
print ("Precision:",metrics.precision_score(y_test,y_pred))
print ("Recall:",metrics.recall_score(y_test,y_pred))
print(metrics.classification_report(y_test, y_pred, labels=[1,0]))

