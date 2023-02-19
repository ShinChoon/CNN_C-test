#ifndef WEIGHTS_BIAS_H
#define WEIGHTS_BIAS_H
#include <stdint.h>
uint8_t weights_map_1[4][1][3][3] = {
    {{{116,105,88},
            {109,109,0},
            {77,74,84}}},
    {{{98,88,88},
            {88,70,84},
            {138,145,70}}},
    {{{77,91,84},
            {95,105,112},
            {112,116,105}}},
    {{{134,159,176},
            {81,74,98},
            {102,91,109}}}
};
uint8_t bias_1[4] = {
    31,
    24,
    28,
    31,
};
uint8_t weights_map_2[8][4][3][3] = {
    {{{74,77,0},
            {155,74,0},
            {67,173,88}},
        {{152,84,84},
            {169,84,95},
            {152,166,134}},
        {{91,84,67},
            {166,134,84},
            {138,134,0}},
        {{67,88,95},
            {145,145,67},
            {67,152,138}}},
    {{{67,88,131},
            {0,84,152},
            {74,0,166}},
        {{166,141,162},
            {134,145,138},
            {138,84,0}},
        {{0,81,145},
            {77,109,152},
            {91,77,176}},
        {{141,155,138},
            {148,152,134},
            {0,145,116}}},
    {{{152,173,169},
            {91,77,88},
            {131,81,131}},
        {{77,148,141},
            {74,70,84},
            {134,84,77}},
        {{152,134,159},
            {88,84,91},
            {148,148,134}},
        {{102,91,162},
            {112,91,84},
            {159,155,74}}},
    {{{109,70,84},
            {131,145,145},
            {169,131,155}},
        {{91,109,88},
            {70,131,148},
            {134,98,162}},
        {{98,134,67},
            {141,81,148},
            {141,131,145}},
        {{145,145,0},
            {138,145,152},
            {152,98,152}}},
    {{{134,155,162},
            {98,102,91},
            {155,152,0}},
        {{67,145,70},
            {141,77,70},
            {91,74,91}},
        {{0,0,141},
            {77,88,70},
            {166,159,138}},
        {{91,95,88},
            {77,98,81},
            {166,70,145}}},
    {{{141,138,148},
            {148,77,152},
            {88,67,81}},
        {{67,141,88},
            {155,131,134},
            {109,74,74}},
        {{148,173,162},
            {138,67,134},
            {88,77,70}},
        {{159,148,155},
            {70,77,112},
            {95,74,77}}},
    {{{0,70,131},
            {95,81,162},
            {91,134,145}},
        {{159,141,155},
            {67,148,91},
            {77,131,81}},
        {{88,84,162},
            {74,145,152},
            {84,134,155}},
        {{131,141,141},
            {148,131,77},
            {0,95,138}}},
    {{{159,77,81},
            {67,95,141},
            {0,162,152}},
        {{131,70,0},
            {98,112,166},
            {77,152,166}},
        {{84,91,134},
            {98,84,159},
            {141,173,152}},
        {{148,131,141},
            {70,155,141},
            {131,77,67}}}
};
uint8_t bias_2[8] = {
    26,
    27,
    30,
    32,
    29,
    31,
    28,
    28,
};
#endif