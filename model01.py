import matplotlib.pyplot as plt
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

#Y
dictY = dict(zip(dataY.index, dataY['rating.band']))
testY = dict(dictY)
ratingBandMapper = {
    'Superb' : 0,
    'Fabulous' : 1,
    'Very Good': 2,
    'Good' : 3,
    'Rating' : 4,
}
for i in dictY:
    dictY[i] = [ratingBandMapper[dictY[i]]]

#data prepared
newDataX = pd.DataFrame.from_dict(dictX, orient='index', columns=['Distance'])
dataX.update(newDataX)
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


#print('Variance score: {}'.format(model.score(x, y)))

# # setting plot style
# plt.style.use('fivethirtyeight')

# # plotting residual errors in training data
# plt.scatter(model.predict(x), model.predict(x) - y,
# 			color = "green", s = 10, label = 'Train data')

# # plotting line for zero residual error
# plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

# ## plotting legend
# plt.legend(loc = 'upper right')

# ## plot title
# plt.title("Residual errors")

# ## method call for showing the plot
# plt.show()