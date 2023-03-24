#ifndef VMM_H
#define VMM_H

#include <stdio.h>
#include <stdint.h>

uint8_t float_int8(float source);
uint8_t float_bin_for_bias_result(float _number);
uint8_t float_bin_for_image_weights(float _number, int isweights);
float bin_float_for_image_weights(uint8_t _number, int isweight);

#endif