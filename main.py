from model01 import model as model_01, testY, dx_list as ls
import numpy as np

def floatToRating(a):
    out = ['Superb',
    'Fabulous',
    'Very Good',
    'Rating',
    'Good']
    b = int(a)
    c = b + 1
    if a - b > c - a :
        a = c 
    else: 
        a = b
    if a > 4 or a <= -1:
        return '''Can't predict'''
    return out[a]

# pre_result = []
# for _, value in testY.items():
#     pre_result.append(value)

# result = []
# for arr in ls:
#     result.append(floatToRating(model_01.predict(np.array([arr])).tolist()[0][0]))

# all = len(pre_result)
# total = 0
# for a, b in zip(pre_result, result):
#     if a == b:
#         total += 1  

# print('ความแม่นยำเท่ากับ {}%'.format(100*total/all))

print('price.from Distance summary.score atmosphere cleanliness facilities location.y security staff valueformoney')
print('Example : \t3300 2.9 9.2 8.9 9.4 9.3 8.9 9.0 9.4 9.4')
test = list(map(float, input('Enter data: \t').split()))
print('Result: {}'.format(floatToRating(model_01.predict(np.array([test])).tolist()[0][0])))
