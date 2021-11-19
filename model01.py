#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#load data
data = pd.read_csv('Hostel.csv')

#data prepare
data.dropna(subset = ["rating.band"], inplace=True)
dataX = data.filter(items=['price.from', 'Distance', 'summary.score', 'atmosphere', 'cleanliness', 'facilities', 'location.y', 'security', 'staff', 'valueformoney'])
dataY = data.filter(items=['rating.band'])

#X
dictX = dict(zip(dataX.index, dataX.Distance))
for i in dictX:
    dictX[i] = [float(dictX[i].replace('km from city centre', ''))]
dictX2 = dict(zip(dataX.index, dataX['price.from']))
for i in dictX2:
    if dictX2[i] < 10000:
        dictX2[i] = [dictX2[i]]
    else:
        dictX2[i] = [dictX2[i] - 1000000]

#Y
dictY = dict(zip(dataY.index, dataY['rating.band']))
testY = dict(dictY)
ratingBandMapper = {
    'Superb' : 4,
    'Fabulous' : 3,
    'Very Good': 2,
    'Good' : 1,
    'Rating' : 0,
}
for i in dictY:
    dictY[i] = [ratingBandMapper[dictY[i]]]

#data prepared
newDataX = pd.DataFrame.from_dict(dictX, orient='index', columns=['Distance'])
newDataX2 = pd.DataFrame.from_dict(dictX2, orient='index', columns=['price.from'])
dataX.update(newDataX)
dataX.update(newDataX2)
dataY = pd.DataFrame.from_dict(dictY, orient='index', columns=['rating.band'])

#train the model
listX = dataX.values.tolist()
listY = dataY.values.tolist()

def classifyData(list):
    ls80 = []
    ls20 = []
    for i in range(len(list)):
        if i <= int(len(list) * 0.8):
            ls80.append(list[i])
        else:
            ls20.append(list[i])
    return (ls80, ls20)

listX_train, listX_test = classifyData(listX)
listY_train, listY_test = classifyData(listY)

xtrain = np.array(listX_train)
ytrain = np.array(listY_train)
model = LinearRegression().fit(xtrain, ytrain)
print(f'R-Squared : {model.score(xtrain, ytrain)}')

#print('Variance score: {}'.format(model.score(xtrain, ytrain)))

# # setting plot style
# plt.style.use('fivethirtyeight')

# # plotting residual errors in training data
# listXPlot = []
# for i in range(len(listX_test)):
#     listXPlot.append(i)
# listXPlot = np.array(listXPlot)
# # plt.scatter(model.predict(xtrain), model.predict(xtrain)-ytrain,
# # 			color = "green", s = 10, label = 'Train data')

# plt.scatter(listXPlot, model.predict(np.array(listX_test)),
# 			color = "blue", s = 10, label = 'Train predicted data')
# # plotting line for zero residual error
# plt.hlines(y = 0, xmin = 0, xmax=100, linewidth = 2)

# ## plotting legend
# plt.legend(loc = 'upper right')

# ## plot title
# plt.title("Residual errors")

# ## method call for showing the plot
# plt.show()