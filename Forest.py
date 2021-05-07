

import pandas as pd
import numpy as np

#loading the dataset
forest = pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/NN/fireforests.csv")
forest=forest.iloc[:,[2,3,4,5,6,7,8,9,10]]

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":forest.columns,"data type(in Python)": forest.dtypes})

#3.	Data Pre-concretecessing
#3.1 Data Cleaning, Feature Engineering, etc
         
          
#details of forest 
forest.info()
forest.describe()          

forest.nunique()

#data types        
forest.dtypes
#checking for na value
forest.isna().sum()
forest.isnull().sum()

#checking unique value for each columns
forest.nunique()

EDA ={"column ": forest.columns,
      "mean": forest.mean(),
      "median":forest.median(),
      "mode":forest.mode(),
      "standard deviation": forest.std(),
      "variance":forest.var(),
      "skewness":forest.skew(),
      "kurtosis":forest.kurt()}

EDA

# covariance for data set 
covariance = forest.cov()
covariance

# Correlation matrix 
Correlation = forest.corr()
Correlation

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
forest.var()                   #area column has low variance 

####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(forest.iloc[:,[0,1,2,3,4,5,6,7]],hue='area')

#boxplot for every columns
forest.columns
forest.nunique()

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(forest.iloc[:,[0,1,2,4,5,6,7]])
df.describe()

##################################
###upport Vector Machines MODEL###
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
num_of_classes = y_test.shape[0]

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =8,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(120,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=64,epochs=10)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set 




