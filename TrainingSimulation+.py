print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_selection import train_test_split

path = '../myData'
data = importDataInfo(path)
data = balanceData(data,display=False)
imagesPath, steerings = loadData(path,data)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))
model = createModel()
model.summary()
history = model.fit(batchGen(xTrain, yTrain, 10, 1),steps_per_epoch=20,validation_data=batchGen(xVal, yVal, 100, 0),epochs=2,validation_steps=200)

