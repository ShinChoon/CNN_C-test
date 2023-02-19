import csv
import pickle
import gzip
import idx2numpy
import numpy as np
import pandas as pd


def float_bin(number, places=6):
    """
    convert the float result into binary string
    e.g. 0.5->"0100000"->32
    param number: float number
    param palces: number of bits
    return int8_t number
    """
    if np.isnan(number):
        number = 0
    source = float("{:.5f}".format(number))
    N_flag = True if source < 0 else False
    # define max and min range
    if abs(source) >= (4-1/2 ** (places-2)):
        source = (4-1/2 ** (places-2))
        source *= -1 if N_flag else 1
    # source = rescaling(source, 4-1/2 ** (places-2))
    result = float_to_int8(source)

    return result


def float_to_int8(source, places=6):
    # to convert float into int8, skip the bianry string part
    source_abs = abs(source)
    answer = 0
    base = 4 - 2**(-places+2)
    resolution = base*2/(2**places-1)
    answer = (int)((source+base)/resolution)
    return answer

def convert_float_int8(number, isweights):
    answer_in = 0
    base = 0.015625
    answer_in = abs((int)(number/base))
    if isweights:
        if(number > 0):
            answer_in = answer_in + 64
        if(number < 0):
            answer_in = answer_in + 128
    return answer_in

def resize_images(data):
    _result = []
    for h in data:
        _reshaped = np.reshape(h, (28, 28))
        _padded = np.pad(_reshaped, pad_width=1)  # to 30, 30
        _result.append(_padded)
        
    return _result

def load_data_shared():
    imagesarray = idx2numpy.convert_from_file("mnist/t10k-images-idx3-ubyte")
    imagesarray = imagesarray/255
    return imagesarray


def converstring_int_list(nu_array):
    num_array = np.array(nu_array)
    num_array = num_array.astype(np.float)
    return num_array

def remove_enpty_space(_list):
    w22=[]
    for ele in _list:
        if ele.strip():
            w22.append(ele)
    return w22


training_data = load_data_shared()
resize_images = resize_images([training_data[0]])

# # prolog
# print('float imagearray[30][30] = {')

# # data
# for row in resize_images:
#     for line in row:
#         print('{')
#         for number in line:
#             number = convert_float_int8(number, False)
#             print('%d, ' % number)
#         print('}, ')

# # epilog
# print('};')

rainfall = pd.read_csv('output/param_decoded.csv', sep=',', header=None)
sized_data = rainfall[0:111]

smessage = sized_data.values

weights_1 = []
weights_map_1 = []

bias_1 = []
bias_map_1 = []

weights_2 = []
sub_weights_map = []
weights_map_2 = []


bias_2 = []
bias_map_2 = []

counter = 0


for i in range(0, 12):
    w = smessage[i][0].split(']')
    w = w[0].split('[')
    ww = remove_enpty_space(w)
    ww = ww[0].split(' ')
    ww = remove_enpty_space(ww)

    weights_1.append(ww)

    if((i+1)%3==0)and(i>0):
        weights_map_1.append(weights_1)
        weights_1 = []

weights_map_1 = converstring_int_list(weights_map_1)
# print("weights_map_1: ", np.shape(weights_map_1))
# prolog

print("#ifndef WEIGHTS_BIAS_H")
print("#define WEIGHTS_BIAS_H")
print("#include <stdint.h>")

print('uint8_t weights_map_1[4][1][3][3] = {')

# data
for map in weights_map_1:
    print('{')
    print('{')
    for line in map:
        print('{')
        for number in line:
            number = convert_float_int8(number, True)
            print('%d, ' % number)
        print('}, ')

    print('},')
    print('},')

# epilog
print('};')

b = smessage[12][0].split(']')
b = b[0].split('[')
bb = remove_enpty_space(b)
bb = bb[0].split(' ')
bias_1 = remove_enpty_space(bb)
bias_1 = converstring_int_list(bias_1)
# print("bias_1: ", np.shape(bias_1))

# prolog
print('uint8_t bias_1[4] = {')

# data
for ele in bias_1:
    ele = float_bin(ele)
    print('%d, ' % ele)
# epilog
print('};')

for i in range(13, 109):
    w = smessage[i][0].split(']')
    w = w[0].split('[')
    ww = remove_enpty_space(w)
    ww = ww[0].split(' ')
    ww = remove_enpty_space(ww)

    weights_2.append(ww)

weights_map_2 =  np.reshape(weights_2,(8,4,3,3))
weights_map_2 = converstring_int_list(weights_map_2)
# print("weights_map_2: ", np.shape(weights_map_2))

# prolog
print('uint8_t weights_map_2[8][4][3][3] = {')

# data
for out_ch in weights_map_2:
    print('{')
    for in_ch in out_ch:
        print('{')
        for line in in_ch:
            print('{')
            for number in line:
                number = convert_float_int8(number, True)
                print('%d, ' % number)
            print('}, ')
        print('}, ')
    print('},')
# epilog
print('};')


for i in range(109, 111):
    b = smessage[i][0].split(']')
    b = b[0].split('[')
    bb = remove_enpty_space(b)
    bb = bb[0].split(' ')
    bb = remove_enpty_space(bb)

    bias_2.append(bb)
for i in bias_2[1]:
    bias_2[0].append(i)
bias_map_2 = bias_2

bias_map_2 = np.reshape(bias_map_2[0], (8))
bias_map_2 = converstring_int_list(bias_map_2)
# print("bias_map_2: ", np.shape(bias_map_2))

# prolog
print('uint8_t bias_2[8] = {')

# data
for ele in bias_map_2:
    ele = float_bin(ele)
    print('%d, ' % ele)
# epilog
print('};')
print("#endif")