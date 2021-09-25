

import pandas as pd
import numpy as np

#loading the dataset
startup = pd.read_csv("E:\\360digiTMG assignment\\Data Science\\NN\\50_Startups (2).csv")

#Rearanging columns
startup=startup.iloc[:,[4,0,1,2]]

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":startup.columns,"data type(in Python)": startup.dtypes})

#3.	Data Pre-concretecessing
#3.1 Data Cleaning, Feature Engineering, etc
         
          
#details of startup 
startup.info()
startup.describe()          

startup.nunique()

#data types        
startup.dtypes
#checking for na value
startup.isna().sum()
startup.isnull().sum()

#checking unique value for each columns
startup.nunique()

EDA ={"column ": startup.columns,
      "mean": startup.mean(),
      "median":startup.median(),
      "mode":startup.mode(),
      "standard deviation": startup.std(),
      "variance":startup.var(),
      "skewness":startup.skew(),
      "kurtosis":startup.kurt()}

EDA

# covariance for data set 
covariance = startup.cov()
covariance

# Correlation matrix 
Correlation = startup.corr()
Correlation

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
startup.var()                   #rain column has low variance 

####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(startup.iloc[:,[0,1,2]],hue='Profit')

#boxplot for every columns
startup.columns
startup.nunique()

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(startup.iloc[:,[0,1,2,3]])
df.describe()

##################################
###support Vector Machines MODEL###
"""5.	Model Building:
5.1	Perform Artificial Neural Network on the given datasets.
5.2	Use TensorFlow keras to build your model in Python and use Neural net package in R
5.3	Briefly explain the output in the documentation for each step in your own words.
5.4	Use different activation functions to get the best model.
"""

# from keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense

from keras.utils import np_utils
# from keras.layers import Dropout,Flatten

np.random.seed(10)

from sklearn.model_selection import train_test_split

model_df_train, model_df_test = train_test_split(df, test_size = 0.2,random_state = 457) # 20% test data
 
x_train = model_df_train.iloc[:,1:].values.astype("float32")
y_train = model_df_train.iloc[:,0].values.astype("float32")
x_test = model_df_test.iloc[:,1:].values.astype("float32")
y_test = model_df_test.iloc[:,0].values.astype("float32")

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =3,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(120,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=25,epochs=5)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set 


