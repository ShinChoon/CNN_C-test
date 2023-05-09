//
// CNN Functions and the CNN Architectrue
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h> //
#include <time.h> //Random seed
#include "cnn.h"

/**
 * @brief Creates a new convolutional layer and returns a pointer to it.
 *
 * @param input_width The width of the input data.
 * @param input_height The height of the input data.
 * @param map_size The size of the convolutional filter.
 * @param input_channels The number of input channels.
 * @param output_channels The number of output channels.
 * @param mode_conv The type of convolution to use (VALID, SAME, or FULL).
 * @return CovLayer* A pointer to the newly created convolutional layer.
 */

CovLayer *InitialCovLayer(int input_width, int input_height, int map_size,
						  int input_channels, int output_channels, int mode_conv)
/*mode_conv = 2: VALID 1: SAME 0: FULL*/
{
	// Allocate memory for the new layer.

	CovLayer *covL = malloc(sizeof(*covL));

	// Set the layer's properties.
	covL->input_height = input_height;
	covL->input_width = input_width;
	covL->map_size = map_size;
	covL->mode_conv = mode_conv;

	covL->input_channels = input_channels;
	covL->output_channels = output_channels;

	covL->is_full_connect = true;

	// Calculate the output dimensions.
	int i, j, c, r;
	int outW = input_width - map_size + 1;
	int outH = input_height - map_size + 1;
	covL->v = calloc(output_channels, sizeof(*(covL->v)));
	for (j = 0; j < output_channels; j++)
	{
		// Allocate memory for the layer's outputs.
		covL->v[j] = calloc(outH, sizeof(*(covL->v)[j]));
		for (r = 0; r < outH; r++)
		{
			covL->v[j][r] = calloc(outW, sizeof(*(covL->v)[j][r]));
		}
	}

	// Return the newly created layer.
	return covL;
}

/**
 * @brief Creates a new pooling layer and returns a pointer to it.
 *
 * @param input_width The width of the input data.
 * @param input_height The height of the input data.
 * @param map_size The size of the pooling filter.
 * @param input_channels The number of input channels.
 * @param output_channels The number of output channels.
 * @param pooling_type The type of pooling to use (MAX or AVG).
 * @return PoolingLayer* A pointer to the newly created pooling layer.
 */
PoolingLayer *InitialPoolingLayer(int input_width, int input_height,
								  int map_size, int input_channels, int output_channels, int pooling_type)
{
	// Allocate memory for the new layer.
	PoolingLayer *poolL = calloc(1, sizeof(*poolL));

	// Set the layer's properties.
	poolL->input_height = input_height;
	poolL->input_width = input_width;
	poolL->map_size = map_size;
	poolL->input_channels = input_channels;
	poolL->output_channels = output_channels;
	poolL->pooling_type = pooling_type;

	// Calculate the output dimensions.
	int outW = input_width / map_size;
	int outH = input_height / map_size;

	// Allocate memory for the layer's outputs.
	int j, r;
	poolL->y = calloc(output_channels, sizeof(*(poolL->y)));
	for (j = 0; j < output_channels; j++)
	{
		poolL->y[j] = calloc(outH, sizeof(*(poolL->y[j])));
		for (r = 0; r < outH; r++)
		{
			poolL->y[j][r] = calloc(outW, sizeof(*(poolL->y[j][r])));
		}
	}

	return poolL;
}

/**
 * @brief Allocates memory for an OutputLayer and initializes its attributes.
 *
 * @param input_num The number of inputs to the layer.
 * @param output_num The number of outputs from the layer.
 * @return A pointer to the newly allocated and initialized OutputLayer.
 */
OutputLayer *InitOutputLayer(int input_num, int output_num)
{
	/*Allocate the memory of OutputLayer and initialize the attributes*/
	OutputLayer *outL = (OutputLayer *)calloc(1, sizeof(*outL));

	outL->input_num = input_num;
	outL->output_num = output_num;

	outL->v = calloc(output_num, sizeof(*(outL->v)));



	outL->is_full_connect = true;

	return outL;
}

/**
 * @brief Applies max pooling to an input matrix and stores the output in an output matrix.
 *
 * @param output A pointer to a 2D output matrix.
 * @param output_size The size of the output matrix.
 * @param input A pointer to a pointer to the input.
 * @param input_size The size of the input matrix.
 * @param map_size The size of the pooling map.
 */
void MaxPooling(uint8_t ***output, MatSize output_size, uint8_t **input,
				MatSize input_size, int map_size)
/*it is just address storage */
{
	int outputW = input_size.columns / map_size;
	int outputH = input_size.rows / map_size;
	
	/* Check if the output size matches the expected dimensions */
	if (output_size.columns != outputW || output_size.rows != outputH)
		printf("[-] ERROR: Output size is wrong! <MaxPooling> \n");

	int i, j, m, n;
	for (i = 0; i < outputH; i++)
	{

		for (j = 0; j < outputW; j++)
		{
			uint8_t *pMax = &(input[0][0]); // initalization!!!
			uint8_t pNumber = 0;
			for (m = i * map_size; m < i * map_size + map_size; m++)
			{

				for (n = j * map_size; n < j * map_size + map_size; n++)
				{

					if (input[m][n] > pNumber)
					{
						pMax = &(input[m][n]);
						pNumber = input[m][n];
					}
				}

				output[i][j] = pMax;
			}
		}
	}
}
