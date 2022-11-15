import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



def welcome():
    print("welcome in salary prediction system")
    print("Please press ENTER key to proceed ")
    input()



def checkcsv():
    csv_files=[]
    current_directory = os.getcwd()
    content=os.listdir(current_directory)

    for file_name in content:
        if file_name.split(".")[-1]=='csv':
              csv_files.append(file_name)
    return csv_files

def check_and_select_csv(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'...',file_name)
        i+=1
    return  csv_files[int(input("select your csv files"))]
    


def graph(x_train,y_train,model,x_test,y_test,y_pred):
    plt.scatter(x_train,y_train,color='red',label='training data')
    plt.plot(x_train,model.predict(x_train),color='blue',label='Best Fit')
    plt.scatter(x_test,y_test,color='green',label='test data')
    plt.scatter(x_test,y_pred,color='black',label='Pred test data')
    plt.title("Salary vs Experience")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()



def main():
    
    welcome()
    try:
       # csv_file=pd.read_csv(r"F:\project\machine learning\attachment_Salary-Data_lyst5512.csv")this is best when file are in same folder
       #print(csv_files)
        csv_files=checkcsv()
        csv_file=check_and_select_csv(csv_files)
        print(csv_file)
        print("csv File is Selected")
        print("Reading csv File......")

        dataset=pd.read_csv(csv_file)
        print("Creating Dataset......")
        x=dataset.iloc[:,:-1].values
        y=dataset.iloc[:,-1].values
        s=float(input("Enter The Test Data Size(between 0 and 1)-->"))

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s)
        print("DataSet is created")
        print("Creating ML Model.....")
        model=LinearRegression()
        model.fit(x_train,y_train)
        print("ML Model is Created")
        print("Press Enter Key TO Check Model in graphical format")
        input()
        y_pred=model.predict(x_test)
        graph(x_train,y_train,model,x_test,y_test,y_pred)
        print("Press ENTER key to check accuracy")
        input()
        accuracy=r2_score(y_pred,y_test)
        print("our Model Accuracy is %2.2f%%"%(accuracy*100))
        print("Our Model Is Ready For Use")
        print(input("Press Enter Key  To Use"))
        print("Experience Year (separated by coma)")

        user=list(float(e) for e in input().split(","))
        ex=[]
        for a in user:
            ex.append([a])
        array=np.array(ex)        
        result=model.predict(array)
        print("Your Salary =")
        result=pd.DataFrame({'Experience':user,'slary':result})
        print(result)
        print("Thankyou For Using Our Ml Model")
        


        
    except FileNotFoundError:
         print("csv file not found")


if __name__=="__main__":
        main()

  
   
