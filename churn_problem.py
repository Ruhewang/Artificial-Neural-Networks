# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras libraries
import keras
from keras.models import Sequential 
from keras.layers import Dense        
from keras.layers import Dropout

classifier=Sequential()    #initialise the NN

#adding the input layer and first hidden layer with dropout
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1)) 

 #adding the second hidden layer
 #no input layer for 2nd hidden layer with dropout
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1)) 

#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
   
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
   
#fitting ANN to training set

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#predict the test set results
y_pre=classifier.predict(X_test)
y_pre=(y_pre>0.5)

#eg problem
"""Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""

#predicting for a single customer
new_pre=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pre=(new_pre>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pre)

"""to prevent overfitting/high varience problem we can use dropout regulisation,where 
    certain neurons of the ann are randomly disabled"""
    
"""tunning the ANN
    Generally all the unchanged values like epochs,batch size(hyperparameters)
    can be tuned in order to obtain higher accuracy"""
    
 from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense  
      
def build_classifier():
    classifier=Sequential()    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn= build_classifier)
parameters={'batch_size':[25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}

grid_search=GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search=grid_search.fit(X_train,y_train)
    best_paramters=grid_search.best_params_
best_accuracy=grid_search.best_score_ 










 





