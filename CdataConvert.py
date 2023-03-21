import csv
import pickle
import gzip
import idx2numpy
import numpy as np
import pandas as pd


def float_bin(number, places=8):
    """
    convert the float result into binary string
    e.g. 0.5->"01 000000"
    param number: float number
    param palces: number of bits
    return int8_t number
    """
    if np.isnan(number):
        number = 0
    source = float("{:.5f}".format(number))
    N_flag = True if source < 0 else False
    # define max and min range
    if abs(source) >= (4-2 ** (-1*places+1)):
        source = (4-2 ** (-1*places+1))
        source *= -1 if N_flag else 1
    # source = rescaling(source, 4-1/2 ** (places-2))
    result = float_to_int8(source)

    return result


def float_to_int8(number, places=8):
    answer_in = 0
    # corresponding to python but leave space for 2 sign bits
    base = 2**(-8+2)
    answer_in = abs((int)(number/base))
    if(number > 0):
        answer_in = answer_in + 64
    if(number < 0):
        answer_in = answer_in + 128
    return answer_in

def convert_float_int8(number, isweights):
    answer_in = 0
    # if isweights:
    base = 2**(-8+2)#corresponding to python but leave space for 2 sign bits
    answer_in = abs((int)(number/base))
    if(number > 0):
        answer_in = answer_in + 64
    if(number < 0):
        answer_in = answer_in + 128
    # else:
    #     base = 2**(-8+1)  # corresponding to python
    #     answer_in = abs((int)(number/base))
    return answer_in

def resize_images(data):
    _reshaped = np.reshape(data, (28, 28))
    _padded = np.pad(_reshaped, pad_width=1)  # to 30, 30
        
    return _padded

def load_data_shared(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding="latin1")
    f.close()

    return test_data


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


def bin_float_for_activation(number):
    sign = (int)(1)
    base = (float)(0.015625) # 2**(-4))
    answer_in = float(0)
    if number > 0:
        number = number-63
    answer_in = (float)(number * base * sign)
    return answer_in


# i = 1
# mini_batch_size = 10
# test_data = load_data_shared()
# testx, texty =  test_data
# take_batch = testx[i*mini_batch_size:(i+1)*mini_batch_size]
# resize_image = resize_images(take_batch[7])
# print("shape: ", np.shape(resize_image))

# # prolog
# print('const uint8_t myimagearray[1][30][30] = {{')

# # data
# for line in resize_image:
#     print('{')
#     for number in line:
#         number = convert_float_int8(number, False)
#         fnumber = bin_float_for_activation(number)
#         print('%f, ' % fnumber)
#     print('}, ')

# # epilog
# print('}};')

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

print('const uint8_t weights_map_1[4][1][3][3] = {')

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
print('const float bias_1[4] = {')

# data
for ele in bias_1:
    # ele = float_bin(ele)
    print('%f, ' % ele)
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
print('const uint8_t weights_map_2[8][4][3][3] = {')

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
print('const float bias_2[8] = {')

# data
for ele in bias_map_2:
    # ele = float_bin(ele)
    print('%f, ' % ele)
# epilog
print('};')
print("#endif")
