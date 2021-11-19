import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def prepareX(data):
    dictX = dict(zip(data.index, data.Distance))
    for i in dictX:
        dictX[i] = [float(dictX[i].replace('km from city centre', ''))]
    dictX2 = dict(zip(data.index, data['price.from']))
    for i in dictX2:
        if dictX2[i] < 10000:
            dictX2[i] = [dictX2[i]]
        else:
            dictX2[i] = [dictX2[i] - 1000000]
    newDataX = pd.DataFrame.from_dict(dictX, orient='index', columns=['Distance'])
    newDataX2 = pd.DataFrame.from_dict(dictX2, orient='index', columns=['price.from'])
    data.update(newDataX)
    data.update(newDataX2)
    
    return data

def prepareY(data):
    dictY = dict(zip(data.index, data['rating.band']))
    ratingBandMapper = {
        'Superb' : 4,
        'Fabulous' : 3,
        'Very Good': 2,
        'Good' : 1,
        'Rating' : 0,
    }
    for i in dictY:
        dictY[i] = [ratingBandMapper[dictY[i]]]
    data = pd.DataFrame.from_dict(dictY, orient='index', columns=['rating.band'])
    
    return data

#load data
data = pd.read_csv('Hostel.csv')

#classify train and test
data.dropna(subset = ["rating.band"], inplace=True)

train_data = data.sample(frac=0.80)

test_data = data
for i in train_data['Unnamed: 0'].to_list():
    test_data = test_data[test_data['Unnamed: 0'] != i]

#data prepare
target = ['price.from', 'Distance', 'summary.score', 'atmosphere', 'cleanliness', 'facilities', 'location.y', 'security', 'staff', 'valueformoney']
train_data_x = train_data.filter(items=target)
train_data_y= train_data.filter(items=['rating.band'])

test_data_x = test_data.filter(items=target)
test_data_y= test_data.filter(items=['rating.band'])

#x
train_data_x = prepareX(train_data_x)
test_data_x = prepareX(test_data_x)

#y
train_data_y = prepareY(train_data_y)
test_data_y = prepareY(test_data_y)

#train
xtrain = np.array(train_data_x)
ytrain = np.array(train_data_y)
model = LinearRegression().fit(xtrain, ytrain)
print(f'R-Squared : {model.score(xtrain, ytrain)}')