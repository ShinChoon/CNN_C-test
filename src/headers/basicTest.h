#ifndef _BASICTEST_H
#define _BASICTEST_H
#define IMCrow 36
#define IMCcol 32

#include "cnn.h"

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

float *bias_mapping(CovLayer *cc, int *bias_number);
float **weights_mapping(CovLayer *cc, int *weights_number);
float **inputs_mapping(CovLayer *cc, MnistImage *inputdata, int *VMM_turns);

void _CnnFF(Cnn *cnn, float **input_data);
void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size);
void _ImportCnn(Cnn *cnn, const char *filename);
ImageArray _ReadImages(const char *filename);
const char *getfield(char *line, int num);
void load_weights(FILE *file_point, CovLayer *cc);
void load_bias(FILE *file_point, CovLayer *cc);

void Conv_image(Cnn *cnn, float **input_array, int VMM_turns, int weights_number);
float **MACoperation(float **input_array, float **weight_array, int VMM_turns);
void save_image(int scale, float **image_data, const char *filename);
void read_data(char* address, char* data);
void write_data(char *address, char *data);

#endif