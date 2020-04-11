
#####################################################################
################Converting to miliseconds (time)#####################
#####################################################################

import csv
from datetime import datetime
import time
with open('data/Gowalla_totalCheckins.txt', 'r') as inputf :
  with open('drive/My Drive/IR/gowalla_test.csv', 'w') as csvoutput:
    reader = csv.reader(inputf, delimiter='\t')
    writer = csv.writer(csvoutput)
    for row in reader :
  #     print(row)
      str1 = row[1]
      year = str1[:4]
      month = str1[5:7]
      day = str1[8:10]
      hour = str1[11:13]
      mini = str1[14:16]
      sec = str1[17:19]
      s = day+'/'+month+'/'+year+' '+hour+':'+mini+':'+sec
  #     print(s)
      d = datetime.strptime(s, "%d/%m/%Y %H:%M:%S")
      t = time.mktime(d.timetuple())
      arr = [row[0], t, row[2], row[3], row[4]]
      writer.writerow(arr)
  #     print(year, month, day, hour, mini, sec)


############################################################################
####################Data PreProcessing #####################################
############################################################################

import csv, sys
arr = {}
with open('data/trial.csv', 'r') as inputf :
  reader = csv.reader(inputf)
  for row in reader :
    if int(row[0]) not in arr :
      arr[int(row[0])] = []
    arr[int(row[0])].append(int(row[1]))
    
with open('data/trial.csv', 'a') as outf :
  writer = csv.writer(outf)
  for i in range(1, 49289) :
    if i not in arr :
      continue
    for j in range(1, 10000) :
      if j in arr[i] :
        continue
      temp = [i, j, 0]
      writer.writerow(temp)
      print('\r' + str(i) + ' ' + str(j), end="")
      sys.stdout.flush()


import csv

with open('data/ratings_data.txt','r') as csvinput:
  with open('data/trial.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    reader = csv.reader(csvinput, delimiter=' ')

    for row in reader:
      all = []
      all.append(row[0])
      all.append(row[1])
      all.append(1)

      writer.writerow(all)


########################################################################
######################## Static Model ##################################
########################################################################




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import warnings
warnings.filterwarnings("ignore")

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, SimpleRNN, Reshape, LSTM, Flatten


def return_classes(df):
    le = LabelEncoder()
    le.fit(df)
    classes = le.classes_.tolist()
    return le.transform(df)


data = pd.read_csv("data/trial.csv")


data.columns = ['user_id', 'product_id', 'rating']

# pd.to_numeric(data['user_id'])
data.head()
data.info()


data.iloc[56000]



X_data = data.iloc[:,:-1].values
y_data = data.iloc[:,-1].values



X_train,X_val,y_train,y_val = train_test_split(X_data,y_data,test_size=0.1)


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)

scaler = Normalizer().fit(X_val)
X_val = scaler.transform(X_val)

# X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
# y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
# y_val = np.reshape(y_val, (y_val.shape[0], 1, y_val.shape[1]))


model = Sequential()
model.add(LSTM(4, input_shape=(1,2),return_sequences=True))
model.summary()
model.add(LSTM(4, input_shape=(2,),return_sequences=True))
model.add(LSTM(4, input_shape=(1,2),return_sequences=False))
# model.add(Dense(64, input_shape=(2,), activation='relu'))
model.add(Dropout(0.05))
# model.add(Reshape((1,4)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(2,activation='softmax'))


model.summary()



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(X_train,y_train,epochs=100,validation_data=(X_val,y_val))


ROW = 56000
pred = np.expand_dims(data.iloc[ROW,:-1].values,axis=0)
np.argmax(model.predict(pred))

data.iloc[0,:-1]


#######################################################################################################
############################ Dynamic Model ############################################################
#######################################################################################################


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import warnings
warnings.filterwarnings("ignore")

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, SimpleRNN, Reshape, LSTM, Flatten

def return_classes(df):
    le = LabelEncoder()
    le.fit(df)
    classes = le.classes_.tolist()
    return le.transform(df)


data = pd.read_csv("data/gowalla_test.csv")[:928670]
data1 = pd.read_csv("data/trial.csv")[:928670]
# 928670

data.columns = ['user_id', 'time', 'co-1', 'co-2', 'productid']
data1.columns = ['user_id', 'product_id', 'rating']

# pd.to_numeric(data['user_id'])
data.head()
data.info()

# del data['time']

X_data = data.iloc[:,:-1].values
y_data = data1.iloc[:,-1].values



X_train,X_val,y_train,y_val = train_test_split(X_data,y_data,test_size=0.1)


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)

scaler = Normalizer().fit(X_val)
X_val = scaler.transform(X_val)

# X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1,4),return_sequences=True))
model.summary()
model.add(LSTM(4,return_sequences=True))
model.add(LSTM(4,return_sequences=False))
# model.add(Dense(64, input_shape=(2,), activation='relu'))
model.add(Dropout(0.05))
# model.add(Reshape((1,4)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(2,activation='softmax'))


model.summary()



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(X_train,y_train,epochs=1,validation_data=(X_val,y_val))


#####################################################################################################
###################### Plotting Graphs ##############################################################
#####################################################################################################

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k=20, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
  

from matplotlib import pyplot
  
ndcg_values = []
i_val = []
for i in range (1, 11) :
  ndcg_values.append(ndcg_at_k(X_val[1], i))
  i_val.append(i)
  
pyplot.plot(i_val, ndcg_values, marker='.')
pyplot.show()
#   print(ndcg_at_k(X_val[1], i))



