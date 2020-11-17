
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from  matplotlib import pyplot as plt
import pandas as pd


from tkinter import *

df = pd.read_csv('India-Inflation.csv')
InitialData = df
#-------------------------------------------------------------------------------------------------------------------------------

# Tkinter Declaration

MainWindow = Tk()
MainWindow.title('Recession Prediction')

TopFrame = Frame(MainWindow)
TopFrame.pack()
MidFrame = Frame(MainWindow)
MidFrame.pack()
LowFrame = Frame(MainWindow)
LowFrame.pack()
svrFrame = LabelFrame(MainWindow, text="Using SVM")
svrFrame.pack(fill="both", expand="yes",side = LEFT)
lrFrame = LabelFrame(MainWindow, text="Using Linear Regression")
lrFrame.pack(fill="both", expand="yes",side = RIGHT)

Heading = Label(TopFrame, text="Welcome to Recession Predication\n",bd=5,font = "Helvetica 16 bold italic")
Heading.pack()


def displayData():


        DataWindow = Toplevel()
        DataWindow.title('Data')
        Data = Label(DataWindow, text=InitialData)
        Data.pack()



DataButton = Button(TopFrame,width=50, text="Display Current Dataset",activebackground = 'black',activeforeground = 'white',bd = 4)
DataButton['command'] = displayData
DataButton.pack()

NoOfMonthLable = Label(MidFrame,text="Enter a year after 2019: ",pady=4)

NoOfMonthLable.pack(side=LEFT)

NoOfMonthEntry = Entry(MidFrame)

NoOfMonthEntry.pack(side=RIGHT)





#-----------------------------------------------------------------------------------------------------------------------

def DisplayPredict():

    global df
    df = df[['Value']]

    var=NoOfMonthEntry.get()
    if var == '':
        error = Label(LowFrame, text="Enter a valid number", fg='red')
        error.pack()

    else:
        try:
             forecast_out = int(NoOfMonthEntry.get())-2019
             if forecast_out <= 0:
                 error = Label(LowFrame, text="enter a value greater then 2019", fg='red')
                 error.pack()



        except BaseException:
            error = Label(LowFrame, text="Enter a valid number", fg='red')
            error.pack()


        df['prediction'] = df[['Value']].shift(-forecast_out)
        #print(df.head())
        #print(df.tail())

        X = np.array(df.drop(['prediction'],1))
        X = X[:-forecast_out]
       # print(X)

        y = np.array(df['prediction'])
        y = y[:-forecast_out]


        x_train, x_test,y_train, y_test = train_test_split(X,y,test_size=0.2)
       # print("x train",x_train)
        #print("x_test",x_test)

        svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)
        svr_rbf.fit(x_train,y_train)

        svm_confidence = svr_rbf.score(x_test, y_test)

        lr = LinearRegression()
        lr.fit(x_train,y_train)

        lr_confidence = lr.score(x_test, y_test)

        x_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:]

        lr_prediction = lr.predict(x_forecast)

        svr_prediction = svr_rbf.predict(x_forecast)

#----------------------------------------------------------------------------------------------
    SVM_Confidence = str(svm_confidence)
    svr_Label = Label(svrFrame, text="\nSVM Confidence:"+SVM_Confidence+'\n',font = "Helvetica 9 italic")
    svr_Label.pack()
    for i in range(forecast_out):
       if svr_prediction[i] > 3.22:
        a=str(svr_prediction[i])
        b=str(i+1+2019)
        svr_Lable =Label(svrFrame,text='Prediction for year('+b +'): '+a
                                       +'--Predicted value is greater then standard value(3.22)')
        svr_Lable.pack()
        svr_Lable = Label(svrFrame, text='Thus LESS chances of recession\n', font='Helvetica 9 bold')
        svr_Lable.pack()
       else:
            a = str(svr_prediction[i])
            b = str(i + 1 + 2019)
            svr_Lable = Label(svrFrame, text='Prediction for year(' + b + '): ' + a
                                            + '--Predicted value is less then standard value(3.22)')
            svr_Lable.pack()
            svr_Lable = Label(svrFrame,text='Thus MORE chances of recession\n',font='Helvetica 9 bold')
            svr_Lable.pack()

    LR_Confidence = str(lr_confidence)
    lr_Label = Label(lrFrame,text="\nLR Confidence: "+LR_Confidence+'\n',font = "Helvetica 9 italic")
    lr_Label.pack()
    for i in range(forecast_out):
        if lr_prediction[i] > 3.22:
            a = str(lr_prediction[i])
            b = str(i + 1 + 2019)
            lr_Label = Label(lrFrame, text='Prediction for year(' + b + '): ' + a
                                             + '--Predicted value is greater then standard value(3.22)')
            lr_Label.pack()
            lr_Label = Label(lrFrame, text='Thus LESS chances of recession\n', font='Helvetica 9 bold')
            lr_Label.pack()
        else:
            a = str(lr_prediction[i])
            b = str(i + 1 + 2019)
            lr_Label = Label(lrFrame, text='Prediction for year(' + b + '): ' + a
                                             + '--Predicted value is less then standard value(3.22)')
            lr_Label.pack()
            lr_Label = Label(lrFrame, text='Thus MORE chances of recession\n', font='Helvetica 9 bold')
            lr_Label.pack()
    #---------------------------------------------------------------
    # Declaring Graph

    GraphX = [2016, 2017, 2018,2019]
    GraphY = [4.941, 2.4909, 4.8607,4.5]
    for i in range(forecast_out):
        GraphX.append(forecast_out + 2019)

    for i in range(forecast_out):
        GraphY.append(lr_prediction[i])

    GraphY1 = [4.941, 2.4909, 4.8607, 4.5]


    for i in range(forecast_out):
        GraphY1.append(svr_prediction[i])

    plt.plot(GraphX, GraphY1, label = 'SVM',linestyle= 'dashed')
    plt.plot(GraphX,GraphY, label = 'LR')
    plt.xlabel('Years')
    plt.ylabel('SVM/LR prediction')
    plt.title('SVR/LR VS. YEAR')
    plt.legend()
    plt.show()




#---------------------------------------------------------------------------------------------------








PredictButton = Button(LowFrame,text="Predict",width=50,command=DisplayPredict, activebackground = 'black',activeforeground = 'white',bd = 4)
PredictButton.pack(side=RIGHT)

MainWindow.mainloop()
