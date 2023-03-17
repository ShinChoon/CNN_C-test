#include "vmm.h"

uint8_t float_int8(float source)
{
    uint8_t answer = 0;
    float base = 3.9375;
    float resolution = 0.0625;
    answer = (uint8_t)((source+base)/resolution);
    return answer;
}

uint8_t float_bin_for_bias_result(float _number)
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


float bin_float_for_image_weights(uint8_t number, int isweight)
{
    int sign = 1;
    float base = 0.015265; // 2**(-6))
    float answer_in = 0;
    uint8_t _number = number;
    // if(isweight)
    {
        base = 0.015265; // 2^-6
        if (_number > 128)
        {
            _number -= 128;
            sign = -1;
        }
        else if (_number > 64)
        {
            _number -= 64;
            sign = 1;
        }
        else
            sign = 0;
    }
    // else
        // base = 0.0078125; // 2**(-7)

    answer_in = (float)(_number * base * sign);
    return answer_in;
}

