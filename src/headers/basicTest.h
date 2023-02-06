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
    float ***(*MACoperation)(float ***, float ***, int, int);
} VMM;

struct weight_map{
    float weights[4][9];
};


VMM *initializeVMM(Cnn *cnn);

float *bias_mapping(CovLayer *cc, int *bias_number);
float ***weights_mapping(CovLayer *cc, int *weights_number, int Scaling);
float ***generate_1d_input(CovLayer *cc, MnistImage **inputimages, int *VMM_turns, int scaling);
void inputs_mapping(MnistImage **images, float **maplist, int *VMM_turns, int scaling, int part_index);
void assign_to_sub_array(float **maplist, float *temp_input, int size_xx, int count_y);

int converfloat_int8(float number, int isweight);
void _CnnFF(CovLayer *conv_layer, PoolingLayer *pool_layer, float **input_data);
void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size);
void _ImportCnn(Cnn *cnn);

void freeConvLayer(CovLayer *cnn);

ImageArray _ReadImages(const char *filename);
const char *getfield(char *line, int num);
void load_weights(CovLayer *cc, float ****weights);
void load_bias(CovLayer *cc, float *bias);

void Conv_image(CovLayer *conv_layer, PoolingLayer *pool_layer, float ***input_array, int VMM_turns, int weights_number, int scaling);
float ***MACoperation(float ***input_array, float ***weight_array, int VMM_turns, int Scaling);
void save_image(int scale, float **image_data, const char *filename);
void read_data(char* address, char* data);
void write_data(char *address, char *data);

MnistImage* Output_image(int cols, int rows, float** imagedata);

#endif