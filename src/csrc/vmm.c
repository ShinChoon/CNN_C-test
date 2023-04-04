#include "vmm.h"

uint8_t float_int8(float source)
{
    uint8_t answer = 0;
    float base = 3.9375;
    float resolution = 0.25;
    float sum = source + base;
    answer = (uint8_t)round((sum) / resolution);
    
    return answer;
}

uint8_t float_bin_for_result(float _number)
/*take the functionality of quantization in RELU*/
{
    float number = _number;
    if (number >= 3.9375) // 4-2**(-4)
        number = 3.9375;
    else if (number <= -3.9375)
        number = -3.9375;

    uint8_t result = float_int8(number);
    return result;
}

float bin_float_for_result(uint8_t _number)
/*take the functionality of quantization in RELU*/
{
    uint8_t number = _number;
    if (number >= 32) // 4-2**(-4)
        number = 32;
    else if (number <= 0)
        number = 0;

    float result = number*0.25-3.9375;
    return result;
}

float bin_float_for_image_weights(uint8_t number, int isweight)
{
    int sign = 1;
    float base = 0.015625; // 2**(-6)) default for weights, because its absolute value range is (0,1)
    float answer_in = 0;
    uint8_t _number = number;
    if (isweight)
    {
        if (_number > 128)
        {
            sign = -1;
            _number -= 128;
        }
        else if (_number > 64) // tested with weight precision
        {
            sign = 1;
            _number -= 64;
        }
        else
            sign = 0;
        answer_in = (float)(_number * base * sign);
    }
    else
    {
        base = 0.015625; // 2**(-8)
        answer_in = (float)(_number) * base * sign;
    }

    return answer_in;
}

float bin_float_for_bias(int8_t number){
    int sign = 1;
    float base = 0.015625; // 2**(-6)) default for weights, because its absolute value range is (0,1)
    float answer_in = 0;
    int8_t _number = number;
    base = 0.015625; // 2**(-8)
    answer_in = (float)(_number)*base * sign;

    return answer_in;
}
