import numpy as np

from model02 import model, test_data_x, test_data_y

out = ['Rating',
    'Good',
    'Very Good',
    'Fabulous',
    'Superb']

def floatToRating(a):
    b = int(a)
    c = b + 1
    if a - b > c - a :
        a = c 
    else: 
        a = b
    if a > 4 or a <= -1:
        return '''Can't predict'''
    return out[a]

total = 0
x_list = model.predict(np.array(test_data_x))
y_list = test_data_y.values.tolist()

for x, y in zip(x_list, y_list):
    if floatToRating(x[0]) == out[y[0]]:
        total += 1

print('ความแม่นยำ : {}'.format(100*total/len(x_list)))