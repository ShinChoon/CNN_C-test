#ifndef CNN_H_
#define CNN_H_

#include <stdbool.h>
#include <stdint.h>
#include "mat.h"
#include "mnist.h"

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
	uint8_t ****map_data;
	uint8_t ****dmap_data;

	uint16_t *basic_data;
	bool is_full_connect;
	bool *connect_model;

	uint8_t ***v;
	uint8_t ***y;

	uint8_t ***d;
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
	uint8_t *basic_data;

	uint8_t ****y;
	uint8_t ***d;
} PoolingLayer;

// Define Output layer
typedef struct OutputLayer
{
	int input_num;
	int output_num;

	uint8_t **wData;
	uint8_t *basic_data;

	uint8_t *v;
	uint8_t *y;
	uint8_t *d;

	bool is_full_connect;
} OutputLayer;

// Define CNN Architectrue
typedef struct ConvolutionalNeuralNetwork
{
	int layer_num;	  // layer_num
	CovLayer *C1;	  // Cov Layer1
	PoolingLayer *S2; // Pooling Layer2
	CovLayer *C3;	  // Cov Layer3
	PoolingLayer *S4; // Pooling Layer4
	OutputLayer *O5;  // Output Layer5
	OutputLayer *O6;  // Output Layer5

	uint8_t *e;
	uint8_t *L;
} Cnn;

// Train Options
typedef struct TrainOptions
{
	int numepochs;
	uint8_t alpha;
} TrainOptions;

CovLayer *InitialCovLayer(int input_width, int input_height, int map_size,
						  int input_channels, int output_channels, int mode_conv);

PoolingLayer *InitialPoolingLayer(int input_width, int inputHeigh, int map_size,
								  int input_channels, int output_channels, int pooling_type);

OutputLayer *InitOutputLayer(int input_num, int output_num);


void MaxPooling(uint8_t ***output, MatSize outputSize, uint8_t **input,
				MatSize inputSize, int map_size);

#endif
