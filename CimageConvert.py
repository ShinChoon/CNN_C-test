import csv
import pickle
import gzip
import idx2numpy
import numpy as np
import pandas as pd


def float_bin(number, places=2):
    """
    convert the float result into binary string
    e.g. 0.5->"01 000000"
    param number: float number
    param palces: number of bits
    return int8_t number
    """
    if np.isnan(number):
        number = 0
    # # define max and min range
    # if number > (2**2-1)/2**2:
    #     number = (2**2-1)/2**2
    # if number < -1 * (2**2-1)/2**2:
    #     number = -1 * (2**2-1)/2**2
    # source = float("{:.5f}".format(number))
    # result = float_to_int8(source, places=places)
    if number > (2**6-1)/2**6:
        number = (2**6-1)/2**6
    if number < -1 * (2**6-1)/2**6:
        number = -1 * (2**6-1)/2**6
    resolution = 2**(-6)  # corresponding to python
    result = round(number/resolution)
    return result


def float_to_int8(number, places=2):
    answer_in = 0
    # corresponding to python but leave space for 2 sign bits
    resolution = 2**(-1*places)
    answer_in = round(number/resolution)
    return answer_in


def convert_float_int8(number, isweights):
    answer_in = 0
    if isweights:
        if number > (2**(8-2)-1)/2**6 :
                number = (2**(8-2)-1)/2**6
        if number < -1 * (2**(8-2)-1)/2**6:
                number = -1 * (2**(8-2)-1)/2**6

        resolution = 2**(-8+2)#sign bits by 2 + value range by 4 
        answer_in = abs(round(number/resolution))
        if(number > 0):
            answer_in = answer_in + 64
        if(number < 0):
            answer_in = answer_in + 128
    else:
        if number > (2**6-1)/2**6:
            number = (2**6-1)/2**6
        if number < -1 * (2**6-1)/2**6:
            number = -1 * (2**6-1)/2**6
        resolution = 2**(-6)  # corresponding to python
        answer_in = abs(round(number/resolution))
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


i = 1
mini_batch_size = 10
test_data = load_data_shared()
testx, texty =  test_data
take_batch = testx[i*mini_batch_size:(i+2)*mini_batch_size]
resize_image = resize_images(take_batch[14])

print("#ifndef _IMAGE_H")
print("#define _IMAGE_H")
print("#include <stdint.h>")

# prolog
print('const uint8_t myimagearray[1][30][30] = {{')

# data
for line in resize_image:
    print('{')
    for number in line:
        number = convert_float_int8(number, False)
        # fnumber = bin_float_for_activation(number)
        print('%s, ' % number)
    print('}, ')

# epilog
print('}};')
print("#endif")