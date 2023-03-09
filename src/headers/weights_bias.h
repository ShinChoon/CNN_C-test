#ifndef WEIGHTS_BIAS_H
#define WEIGHTS_BIAS_H
#include <stdint.h>
uint8_t weights_map_1[4][1][3][3] = {
    {
        {
            {98,90,82},
            {71,71,71},
            {188,154,143}
        },
    },
    {
        {
            {105,105,98},
            {105,101,86},
            {75,71,143}
        },
    },
    {
        {
            {98,90,101},
            {109,86,94},
            {146,146,146}
        },
    },
    {
        {
            {67,94,98},
            {82,109,116},
            {135,116,120}
        },
    },
};
uint8_t bias_1[4] = {32,31,22,31};
uint8_t weights_map_2[8][4][3][3] = {
    {
        {
            {165,139,162},
            {162,79,143},
            {158,86,94}
        },
        {
            {0,82,131},
            {86,143,131},
            {94,150,158}
        },
        {
            {158,143,75},
            {131,67,154},
            {82,131,86}
        },
        {
            {90,98,143},
            {75,135,158},
            {90,143,158}
        },
    },
    {
        {
            {131,169,143},
            {143,143,146},
            {135,154,75}
        },
        {
            {131,90,79},
            {139,0,146},
            {75,86,150}
        },
        {
            {158,86,139},
            {146,90,67},
            {158,75,135}
        },
        {
            {94,67,135},
            {94,67,162},
            {94,139,150}
        },
    },
    {
        {
            {75,139,169},
            {139,109,143},
            {82,98,82}
        },
        {
            {165,158,154},
            {101,135,158},
            {82,90,79}
        },
        {
            {139,94,135},
            {75,139,143},
            {79,120,131}
        },
        {
            {150,150,146},
            {0,135,158},
            {75,86,90}
        },
    },
    {
        {
            {98,135,131},
            {75,0,146},
            {150,150,131}
        },
        {
            {75,86,94},
            {139,139,67},
            {150,162,143}
        },
        {
            {71,94,101},
            {71,131,0},
            {131,131,158}
        },
        {
            {79,98,71},
            {169,162,135},
            {135,139,158}
        },
    },
    {
        {
            {131,154,75},
            {154,90,79},
            {82,150,67}
        },
        {
            {158,139,139},
            {154,158,143},
            {105,75,105}
        },
        {
            {75,143,158},
            {67,139,109},
            {79,90,101}
        },
        {
            {158,131,177},
            {86,158,150},
            {86,79,86}
        },
    },
    {
        {
            {139,150,94},
            {101,75,79},
            {75,0,105}
        },
        {
            {154,162,165},
            {90,79,101},
            {135,143,131}
        },
        {
            {150,154,158},
            {116,109,94},
            {135,71,98}
        },
        {
            {71,67,158},
            {79,82,79},
            {154,150,135}
        },
    },
    {
        {
            {158,94,143},
            {0,71,113},
            {94,90,150}
        },
        {
            {71,146,71},
            {75,113,82},
            {131,131,150}
        },
        {
            {162,90,131},
            {90,94,71},
            {101,0,67}
        },
        {
            {0,67,75},
            {131,0,139},
            {158,165,146}
        },
    },
    {
        {
            {150,71,135},
            {135,90,131},
            {158,67,86}
        },
        {
            {90,71,150},
            {86,71,154},
            {79,154,0}
        },
        {
            {79,131,150},
            {79,146,75},
            {90,0,75}
        },
        {
            {90,162,135},
            {90,165,135},
            {139,154,0}
        },
    },
};
uint8_t bias_2[8] = {28,23,30,31,31,29,28,29};
#endif