#ifndef _BASICTEST_H
#define _BASICTEST_H
#define IMCrow 36
#define IMCcol 32
#include <stdlib.h>
#include "cnn.h"

typedef struct VMMStructure
{
    int cols;
    int rows;
    Cnn *Cnn;
    uint8_t ***(*MACoperation)(uint8_t ***, uint8_t ***, int, int);
} VMM;

struct weight_map
{
    uint8_t weights[4][9];
};

VMM *initializeVMM(Cnn *cnn);
uint8_t *bias_mapping(CovLayer *cc);
uint8_t ***weights_mapping(CovLayer *cc, int *weights_number, int Scaling);
void inputs_mapping(CovLayer *cc, MnistImage **images, uint8_t ***maplist, int *VMM_turns, int scaling, int part_index);
void assign_to_sub_array(uint8_t ***maplist, uint8_t *temp_input, int size_xx, int count_y, int scal);
uint8_t ***generate_input_array();
uint8_t ***generate_result_array();

void freeConvLayer(CovLayer *cnn);
void freePoolLayer(PoolingLayer *pol);

void _CnnFF(CovLayer *conv_layer, PoolingLayer *pool_layer, uint8_t *bias_array);
void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size);
void _ImportCnn(Cnn *cnn);
ImageArray _ReadImages(const char *filename);
const char *getfield(char *line, int num);
void load_weights(CovLayer *cc, uint8_t ****weights);
void load_bias(CovLayer *cc, uint8_t *bias);

void Conv_image(CovLayer *conv_layer, int input_width, uint8_t ***input_array, int VMM_turns, int weights_number, int scaling, int *column_index);
uint8_t ***MACoperation(uint8_t ***input_array, uint8_t ***weight_array, int VMM_turns, int Scaling);
void save_image(int scale, uint8_t **image_data);
void read_data(char *address, char *data);
void write_data(char *address, char *data);
void _VMMMACoperation(uint8_t ***result_list, int pagenumber, int Scaling);

MnistImage *Output_image(int cols, int rows, uint8_t **imagedata);
void _free_3darray(int ***data, size_t xlen, size_t ylen);
uint8_t ***_alloc_3darray(size_t xlen, size_t ylen, size_t zlen);
uint8_t **_alloc_2darray(size_t xlen, size_t ylen);
uint8_t *_alloc_1darray(size_t xlen);

// statically initialize some data in .data section
#endif
