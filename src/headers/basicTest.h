#ifndef _BASICTEST_H
#define _BASICTEST_H

#include "cnn.h"

#define AVG_POOLING 0 // Pooling with average
#define MAX_POOLING 1 // Pooling with Maximum
#define MIN_POOLING 2 // Pooling with Minimum
#define IMCrow 36
#define IMCcol 32

float** weigths_mapping(Cnn *cnn);
float** inputs_mapping(MnistImage* inputdata, MatSize input_size, int* VMM_turns);
void _CnnFF(Cnn *cnn, float **input_data);
void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size);
void _ImportCnn(Cnn *cnn, const char *filename);
ImageArray _ReadImages(const char *filename);
const char *getfield(char *line, int num);
void load_weights(FILE *file_point, CovLayer *cc);
void load_bias(FILE *file_point, CovLayer *cc);

float **MACoperation(float **input_array, float **weight_array, int VMM_turns);

typedef struct VMMStructure
{
    int cols;
    int rows;
    Cnn *Cnn;
    float **(*MACoperation)(float **, float **, int);
} VMM;

struct weight_map{
    float weights[4][9];
};

VMM *initializeVMM(Cnn *cnn);
#endif