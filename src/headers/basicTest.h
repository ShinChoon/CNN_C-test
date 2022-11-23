#ifndef _BASICTEST_H
#define _BASICTEST_H

#include <stdbool.h>
#include "mat.h"
#include "mnist.h"

#define AVG_POOLING 0 // Pooling with average
#define MAX_POOLING 1 // Pooling with Maximum
#define MIN_POOLING 2 // Pooling with Minimum
#define IMCrow 36
#define IMCcol 32

#define AVG_POOLING 0 // Pooling with average
#define MAX_POOLING 1 // Pooling with Maximum
#define MIN_POOLING 2 // Pooling with Minimum

// Define Structure of Convolutional Layer
typedef struct ConvolutionalLayer
{
    int input_width;
    int input_height;
    int map_size;
    int mode_conv;

    int input_channels;
    int output_channels;
    float ****map_data;
    float ****dmap_data;

    float *basic_data;
    bool is_full_connect;
    bool *connect_model;

    float ***v;
    float ***y;

    float ***d;
} CovLayer;

// Define pooling layer
typedef struct PoolingLayer
{
    int input_width;
    int input_height;
    int map_size;

    int input_channels;
    int output_channels;

    int pooling_type;
    float *basic_data;

    float ***y;
    float ***d;
} PoolingLayer;

// Define Output layer
typedef struct OutputLayer
{
    int input_num;
    int output_num;

    float **wData;
    float *basic_data;

    float *v;
    float *y;
    float *d;

    bool is_full_connect;
} OutputLayer;

// Define CNN Architectrue
typedef struct ConvolutionalNeuralNetwork
{
    int layer_num;    // layer_num
    CovLayer *C1;     // Cov Layer1
    PoolingLayer *S2; // Pooling Layer2
    CovLayer *C3;     // Cov Layer3
    PoolingLayer *S4; // Pooling Layer4
    OutputLayer *O5;  // Output Layer5
    OutputLayer *O6;  // Output Layer5

    float *e;
    float *L;
} Cnn;

// Train Options
typedef struct TrainOptions
{
    int numepochs;
    float alpha;
} TrainOptions;

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
float** weights_mapping(Cnn *cnn, int* weights_number);
float** inputs_mapping(MnistImage* inputdata, MatSize input_size, int* VMM_turns);
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

void AvgPooling(float **output, MatSize outputSize, float **input,
                MatSize inputSize, int map_size);

void MaxPooling(float **output, MatSize outputSize, float **input,
                MatSize inputSize, int map_size);

CovLayer *InitialCovLayer(int input_width, int input_height, int map_size,
                          int input_channels, int output_channels, int mode_conv);
void CovLayerConnect(CovLayer *covL, bool *connect_model);

PoolingLayer *InitialPoolingLayer(int input_width, int inputHeigh, int map_size,
                                  int input_channels, int output_channels, int pooling_type);
void PoolingLayerConnect(PoolingLayer *poolL, bool *connect_model);

OutputLayer *InitOutputLayer(int input_num, int output_num);

float ActivationSigma(float input, float bas);
float ActivationReLu(float input, float bas);

void read_data(char* address, char* data);
void write_data(char *address, char *data);

#endif