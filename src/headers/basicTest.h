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
    void (*MACoperation)(uint8_t ***, uint8_t ***, uint8_t ***, int, int);
} VMM;

struct weight_map
{
    uint8_t weights[4][9];
};

VMM *initializeVMM(Cnn *cnn);
void weights_mapping(CovLayer *cc, uint8_t ***VMM_weights_map, int *weights_number, int scaling);
void inputs_mapping(CovLayer *cc, MnistImage **images, uint8_t ***VMM_input_array, int *VMM_turns, 
                    int scaling, int part_index, int scal_index);
void assign_to_sub_array(uint8_t ***maplist, uint8_t *temp_input, int size_xx, int count_y, int scal);
uint8_t ***generate_input_array(int scal, int size);
uint8_t ***generate_result_array(int scal, int VMM_turns);

void freeConvLayer(CovLayer *cnn);
void freePoolLayer(PoolingLayer *pol);

void _CnnFF(CovLayer *conv_layer, PoolingLayer *pool_layer);
void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size, int i);
void _ImportCnn(Cnn *cnn, int i);
ImageArray _ReadImages(const char *filename);
const char *getfield(char *line, int num);
void load_weights(CovLayer *cc, uint8_t ****weights);
void load_bias(CovLayer *cc, uint8_t *bias);

void Conv_imageConv_image(CovLayer *conv_layer, PoolingLayer *pool_layer, uint8_t ***input_array,
                          int VMM_turns, int weights_number, int scaling, int *column_index);
void MACoperation(uint8_t ***input_array, uint8_t ***output_array, uint8_t ***weight_array, 
                    int VMM_turns, int scaling);
void save_image(int scale, uint8_t **image_data);
void read_data(char *address, char *data);
void write_data(char *address, char *data);
void _VMMMACoperation(uint8_t ***result_list, int pagenumber, int Scaling);

ImageArray Output_image(int cols, int rows, uint8_t ***imagedata, int number);
uint8_t ***alloc_3darray(size_t xlen, size_t ylen, size_t zlen);
uint8_t **alloc_2darray(size_t xlen, size_t ylen);
uint8_t *alloc_1darray(size_t xlen);
void free_3darray(uint8_t ***data, size_t xlen, size_t ylen);
void free_2darray(uint8_t **data, size_t xlen);
// statically initialize some data in .data section
#endif
