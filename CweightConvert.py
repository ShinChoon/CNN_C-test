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

def param_extraction():
    rainfall = pd.read_csv('output/param_decoded.csv', sep=',', header=None)
    sized_data = rainfall

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

    w_fc1_group = []
    weights_fc1 = []
    bias_fc_1 = []

    w_fc2_group = []
    weights_fc2 = []
    bias_fc_2 = []



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

    b = smessage[12][0].split(']')
    b = b[0].split('[')
    bb = remove_enpty_space(b)
    bb = bb[0].split(' ')
    bias_1 = remove_enpty_space(bb)
    bias_map_1 = converstring_int_list(bias_1)

    for i in range(13, 109):
        w = smessage[i][0].split(']')
        w = w[0].split('[')
        ww = remove_enpty_space(w)
        ww = ww[0].split(' ')
        ww = remove_enpty_space(ww)

        weights_2.append(ww)

    weights_map_2 =  np.reshape(weights_2,(8,4,3,3))
    weights_map_2 = converstring_int_list(weights_map_2)


    for i in range(109, 111):
        b = smessage[i][0].split(']')
        b = b[0].split('[')
        bb = remove_enpty_space(b)
        bb = bb[0].split(' ')
        bb = remove_enpty_space(bb)
        for ele in bb:
            bias_2.append(ele)

    bias_map_2 = converstring_int_list(bias_2)

    for index in range(288):
        weights_fc1 = []
        for i in range(111+index*5, 116+index*5):
            w = smessage[i][0].split(']')
            w = w[0].split('[')
            ww = remove_enpty_space(w)
            ww = ww[0].split(' ')
            ww = remove_enpty_space(ww)
            for ele in ww:
                weights_fc1.append(ele)


        w_fc1_group.append(weights_fc1)

    w_fc1_group = converstring_int_list(w_fc1_group)

    for i in range(1551, 1556):
        b = smessage[i][0].split(']')
        b = b[0].split('[')
        bb = remove_enpty_space(b)
        bb = bb[0].split(' ')
        bb = remove_enpty_space(bb)
        for ele in bb:
            bias_fc_1.append(ele)

    bias_fc_1 = converstring_int_list(bias_fc_1)


    for index in range(32):
        weights_fc2 = []
        for i in range(1556+index*2, 1558+index*2):
            w = smessage[i][0].split(']')
            w = w[0].split('[')
            ww = remove_enpty_space(w)
            ww = ww[0].split(' ')
            ww = remove_enpty_space(ww)
            for ele in ww:
                weights_fc2.append(ele)

        w_fc2_group.append(weights_fc2)

    w_fc2_group = converstring_int_list(w_fc2_group)
    for i in range(1619, 1621):
        # print("smessage[i][0]: ", smessage[i][0])
        b = smessage[i][0].split(']')
        b = b[0].split('[')
        bb = remove_enpty_space(b)
        bb = bb[0].split(' ')
        bb = remove_enpty_space(bb)
        for ele in bb:
            bias_fc_2.append(ele)

    bias_fc_2 = converstring_int_list(bias_fc_2)


    params = {
        "weights_map_1": weights_map_1,
        "bias_map_1": bias_map_1,
        "weights_map_2": weights_map_2,
        "bias_map_2": bias_map_2,
        "w_fc1_group": w_fc1_group,
        "bias_fc_1": bias_fc_1,
        "w_fc2_group": w_fc2_group,
        "bias_fc_2": bias_fc_2

    }

    return params

params = param_extraction()

print("#ifndef WEIGHTS_BIAS_H")
print("#define WEIGHTS_BIAS_H")
print("#include <stdint.h>")

print('const uint8_t weights_map_1[4][1][3][3] = {')

# data
for map in params["weights_map_1"]:
    print('{')
    print('{')
    for line in map:
        print('{')
        for number in line:
            number = convert_float_int8(number, True)
            print('%s, ' % number)
        print('}, ')

    print('},')
    print('},')

# epilog
print('};')

# print("bias_1: ", np.shape(bias_1))

# prolog
print('const int8_t bias_1[4] = {')

# data
for ele in params["bias_map_1"]:
    ele = float_bin(ele)
    print('%s, ' % ele)
# epilog
print('};')

# print("weights_map_2: ", np.shape(weights_map_2))

# prolog
print('const uint8_t weights_map_2[8][4][3][3] = {')

# data
for out_ch in params["weights_map_2"]:
    print('{')
    for in_ch in out_ch:
        print('{')
        for line in in_ch:
            print('{')
            for number in line:
                number = convert_float_int8(number, True)
                print('%s, ' % number)
            print('}, ')
        print('}, ')
    print('},')
# epilog
print('};')




# prolog
print('const int8_t bias_2[8] = {')

# data
for ele in params["bias_map_2"]:
    ele = float_bin(ele)
    print('%s, ' % ele)
# epilog
print('};')


# data
print('const uint8_t weights_map_3[288][32] = {')
for out_ch in params["w_fc1_group"]:
    if len(out_ch)<32:
        print("len: @@@@@   ", len(out_ch))
    print('{')
    for number in out_ch:
        number = convert_float_int8(number, True)
        print('%s, ' % number)
    print('}, ')
# epilog
print('};')


# prolog
print('const int8_t bias_3[32] = {')
if len(params["bias_fc_1"]) < 32:
    print("len: @@@@@   ", len(params["bias_fc_1"]))

# data
for ele in params["bias_fc_1"]:
    ele = float_bin(ele)
    print('%s, ' % ele)
# epilog
print('};')



# data
print('const uint8_t weights_map_4[32][10] = {')
for out_ch in params["w_fc2_group"]:
    if len(out_ch) < 10:
        print("len: @@@@@   ", len(out_ch))
    print('{')
    for number in out_ch:
        number = convert_float_int8(number, True)
        print('%s, ' % number)
    print('}, ')
# epilog
print('};')



# prolog
print('const int8_t bias_4[10] = {')

# data
for ele in params["bias_fc_2"]:
    if len(params["bias_fc_2"]) < 10:
        print("len: @@@@@   ", len(params["bias_fc_2"]))
    ele = float_bin(ele)
    print('%s, ' % ele)
# epilog
print('};')
print("#endif")
