from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

main = tkinter.Tk()
main.title("Thyroid Disease Classification Using Machine Learning Algorithms") 
main.geometry("1300x1200")

global dataset, X, Y, X_train, y_train, X_test, y_test
global accuracy, precision, recall, fscore, labels, rf
global scaler, label_encoder

def loadData():
    global dataset, labels 
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    labels = np.unique(dataset['Class'])
    label = dataset.groupby('Class').size()
    
    label.plot(kind="bar")
    plt.xlabel("Thyroid Disease Type")
    plt.ylabel("Count")
    plt.title("Thyroid Disease Graph")
    plt.show()
    
def datasetProcessing():
    text.delete('1.0', END)
    global dataset, label_encoder, scaler, X, Y, X_train, y_train, X_test, y_test
    dataset.fillna(0, inplace = True)

    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            label_encoder.append(le)
    text.insert(END,"Dataset After Preprocessing\n\n")
    text.insert(END,str(dataset)+"\n\n")

    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype(int)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def runDecisionTree():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, cnn_model
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    calculateMetrics("Decision Tree", y_test, predict)

def runSVM():
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", y_test, predict)

def runRandomForest():
    global rf
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", y_test, predict)

def runMLP():
    mlp = MLPClassifier() 
    mlp.fit(X_train, y_train)
    predict = mlp.predict(X_test)
    calculateMetrics("MLP", y_test, predict)

def graph():
    df = pd.DataFrame([['Decision Tree','Accuracy',accuracy[0]],['Decision Tree','Precision',precision[0]],['Decision Tree','Recall',recall[0]],['Decision Tree','FSCORE',fscore[0]],
                       ['SVM','Accuracy',accuracy[1]],['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','FSCORE',fscore[1]],
                       ['Random Forest','Accuracy',accuracy[2]],['Random Forest','Precision',precision[2]],['Random Forest','Recall',recall[2]],['Random Forest','FSCORE',fscore[2]],
                       ['MLP','Accuracy',accuracy[3]],['MLP','Precision',precision[3]],['MLP','Recall',recall[3]],['MLP','FSCORE',fscore[3]],
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show()

def predictDisease():
    text.delete('1.0', END)
    global scaler, label_encoder, labels, rf
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    data = dataset.values
    columns = dataset.columns
    types = dataset.dtypes.values
    index = 0
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            dataset[columns[i]] = pd.Series(label_encoder[index].transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            index = index + 1
    dataset = dataset.values
    X = scaler.transform(dataset)
    predict = rf.predict(X)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(data[i])+" =====> Predicted As "+str(labels[predict[i]])+"\n\n")
    text.insert(END, "   You can consult:\n\n   Dr Rabinder Nath Mehrotra\n   Endocrinology | 20 years exp\n   Apollo Health City Jubilee Hills\n   MBBS; MD (Internal Medicine); DM (Endocrinology); DNB (Endocrinology)\n\n")
    text.insert(END, "   Dr Ravisankar Erukulpati\n   Endocrinology, Transplant Surgery | 18 years exp\n   Apollo Health City Jubilee Hills\n   Apollo Sugar Clinic Jubileehills\n   MBBS; MRCP (UK), CCT Diabetes & Endocrinology (UK)\n\n")

    

font = ('times', 16, 'bold')
title = Label(main, text='Thyroid Disease Classification Using Machine Learning Algorithms')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Thyroid Disease Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=datasetProcessing)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=310,y=150)
dtButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=610,y=150)
svmButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRandomForest)
rfButton.place(x=860,y=150)
rfButton.config(font=font1)

mlpButton = Button(main, text="Run MLP Algorithm", command=runMLP)
mlpButton.place(x=50,y=200)
mlpButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=310,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Thyroid from Test Data", command=predictDisease)
predictButton.place(x=610,y=200)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
