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
    'Rating' : 3,
    'Good' : 4
}
for i in dictY:
    dictY[i] = [ratingBandMapper[dictY[i]]]

#data prepared
newDataX = pd.DataFrame.from_dict(dictX, orient='index', columns=['Distance'])
dataX.update(newDataX)
dataY = pd.DataFrame.from_dict(dictY, orient='index', columns=['rating.band'])

#train the model
dx_list = dataX.values.tolist()
dy_list = dataY.values.tolist()
x = np.array(dx_list)
y = np.array(dy_list)
model = LinearRegression().fit(x, y)

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