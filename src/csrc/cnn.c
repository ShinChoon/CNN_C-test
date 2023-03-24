//
// CNN Functions and the CNN Architectrue
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h> //
#include <time.h> //Random seed
#include "cnn.h"

void CnnSetup(Cnn *cnn, MatSize input_size, int output_size)
{
	int map_size = 3;
	cnn->layer_num = 5; // layers = 5
	int pool_scale = 2;

	MatSize temp_input_size;

	// Layer1 Cov input size: {28,28}
	temp_input_size.columns = input_size.columns;
	temp_input_size.rows = input_size.rows;
	cnn->C1 = InitialCovLayer(temp_input_size.columns,
							  temp_input_size.rows, map_size, 1, 4, SAME);

	// Layer2 Pooling input size: {28,28}
	temp_input_size.columns = temp_input_size.columns - map_size + 1;
	temp_input_size.rows = temp_input_size.rows - map_size + 1;
	cnn->S2 = InitialPoolingLayer(temp_input_size.columns,
								  temp_input_size.rows, pool_scale, 4, 4, MAX_POOLING);

	// Layer3 Cov input size: {14,14}
	temp_input_size.columns = temp_input_size.columns / 2;
	temp_input_size.rows = temp_input_size.rows / 2;
	cnn->C3 = InitialCovLayer(temp_input_size.columns,
							  temp_input_size.rows, map_size, 4, 8, VALID);

	// Layer4 Pooling with average. Input size: {12,12}
	temp_input_size.columns = temp_input_size.columns - map_size + 1;
	temp_input_size.rows = temp_input_size.rows - map_size + 1;
	cnn->S4 = InitialPoolingLayer(temp_input_size.columns,
								  temp_input_size.rows, pool_scale, 8, 8, MAX_POOLING);

	// Layer5 Output layer. Input size: {6,6}
	temp_input_size.columns = temp_input_size.columns / 2;
	temp_input_size.rows = temp_input_size.rows / 2;
	cnn->O5 = InitOutputLayer(temp_input_size.columns * temp_input_size.rows * 32,
							  output_size);

	// Layer6 Output layer. Input size: {6,6}
	temp_input_size.columns = temp_input_size.columns;
	temp_input_size.rows = temp_input_size.rows;
	cnn->O6 = InitOutputLayer(temp_input_size.columns * temp_input_size.rows * 32,
							  output_size);

	cnn->e = calloc(cnn->O6->output_num, sizeof(*(cnn->e)));
}

CovLayer *InitialCovLayer(int input_width, int input_height, int map_size,
						  int input_channels, int output_channels, int mode_conv)
/*
	mode_conv = 2: VALID 1: SAME 0: FULL
*/
{
	CovLayer *covL = malloc(sizeof(*covL));

	covL->input_height = input_height;
	covL->input_width = input_width;
	covL->map_size = map_size;
	covL->mode_conv = mode_conv;

	covL->input_channels = input_channels;
	covL->output_channels = output_channels;

	covL->is_full_connect = true; //
	int i, j, c, r;
	// covL->map_data = malloc(output_channels * sizeof(*covL->map_data));
	// for (i = 0; i < output_channels; i++)
	// {
	// 	covL->map_data[i] = malloc(input_channels * sizeof(*(covL->map_data)[i]));
	// 	for (j = 0; j < input_channels; j++)
	// 	{
	// 		covL->map_data[i][j] = malloc(map_size * sizeof(*(covL->map_data)[i][j]));
	// 		for (r = 0; r < map_size; r++)
	// 		{
	// 			covL->map_data[i][j][r] = malloc(map_size * sizeof(*(covL->map_data)[i][j][r]));
	// 		}
	// 	}
	// }

	// covL->basic_data = malloc(output_channels * sizeof(*(covL->basic_data)));

	int outW = input_width - map_size + 1;
	int outH = input_height - map_size + 1;
	// covL->d = (uint8_t ***)malloc(output_channels * sizeof(uint8_t **));
	covL->v = calloc(output_channels, sizeof(*(covL->v)));
	// covL->y = malloc(output_channels * sizeof(*(covL->v)));
	for (j = 0; j < output_channels; j++)
	{
		// covL->d[j] = (uint8_t **)malloc(outH * sizeof(uint8_t *));
		covL->v[j] = calloc(outH, sizeof(*(covL->v)[j]));
		// covL->y[j] = malloc(outH * sizeof(*(covL->v)[j]));
		for (r = 0; r < outH; r++)
		{
			// covL->d[j][r] = (uint8_t *)calloc(outW, sizeof(uint8_t));
			covL->v[j][r] = calloc(outW, sizeof(*(covL->v)[j][r]));
			// covL->y[j][r] = malloc(outW * sizeof(*(covL->v)[j][r]));
		}
	}

	return covL;
}

PoolingLayer *InitialPoolingLayer(int input_width, int input_height,
								  int map_size, int input_channels, int output_channels, int pooling_type)
{
	PoolingLayer *poolL = calloc(1, sizeof(*poolL));

	poolL->input_height = input_height;
	poolL->input_width = input_width;
	poolL->map_size = map_size;
	poolL->input_channels = input_channels;
	poolL->output_channels = output_channels;
	poolL->pooling_type = pooling_type;

	// poolL->basic_data = malloc(output_channels * sizeof(*(poolL->basic_data)));

	int outW = input_width / map_size;
	int outH = input_height / map_size;

	int j, r;
	// poolL->d = (uint8_t ***)malloc(output_channels * sizeof(uint8_t **));
	poolL->y = calloc(output_channels, sizeof(*(poolL->y)));
	for (j = 0; j < output_channels; j++)
	{
		// poolL->d[j] = (uint8_t **)malloc(outH * sizeof(uint8_t *));
		poolL->y[j] = calloc(outH, sizeof(*(poolL->y[j])));
		for (r = 0; r < outH; r++)
		{
			// poolL->d[j][r] = (uint8_t *)calloc(outW, sizeof(uint8_t));
			poolL->y[j][r] = calloc(outW, sizeof(*(poolL->y[j][r])));
		}
	}

	return poolL;
}

OutputLayer *InitOutputLayer(int input_num, int output_num)
{
	OutputLayer *outL = (OutputLayer *)malloc(sizeof(*outL));

	outL->input_num = input_num;
	outL->output_num = output_num;

	outL->basic_data = (uint8_t *)calloc(output_num, sizeof(uint8_t));

	outL->d = malloc(output_num * sizeof(uint8_t));
	outL->v = malloc(output_num * sizeof(uint8_t));
	outL->y = malloc(output_num * sizeof(uint8_t));

	//
	outL->wData = (uint8_t **)malloc(output_num * sizeof(uint8_t *)); //

	srand((unsigned)time(NULL));
	for (int i = 0; i < output_num; i++)
	{
		outL->wData[i] = (uint8_t *)malloc(input_num * sizeof(uint8_t));
		// for (int j = 0; j < input_num; j++)
		// {
		// uint8_t randnum = (((uint8_t)rand() / (uint8_t)RAND_MAX) - 0.5) * 2; //
		// outL->wData[i][j] = randnum * sqrt((uint8_t)6.0 / (uint8_t)(input_num + output_num));
		// }
	}

	outL->is_full_connect = true;

	return outL;
}

int vecmaxIndex(uint8_t *vec, int veclength)
{
	int i;
	uint8_t maxnum = 0;
	int maxIndex = 0;
	for (i = 0; i < veclength; i++)
	{
		if (maxnum < vec[i])
		{
			maxnum = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

uint8_t CnnTest(Cnn *cnn, ImageArray input_data,
				LabelArray output_data, int test_num)
{
	int n = 0;
	int incorrect_num = 0;

	for (n = 0; n < test_num; n++)
	{
		printf("=== Testing ... : %d/%d%c", n, test_num, (char)13);

		CnnFF(cnn, input_data->image_point[n].image_data);

		if (vecmaxIndex(cnn->O6->y, cnn->O6->output_num) !=
			vecmaxIndex(output_data->label_point[n].LabelData, cnn->O6->output_num))
		{
			incorrect_num++;
		}
		CnnClear(cnn);
	}
	return (uint8_t)incorrect_num / (uint8_t)test_num;
}

// Save CNN
void SaveCnn(Cnn *cnn, const char *filename)
{
	FILE *file_point = NULL;
	file_point = fopen(filename, "wb");
	if (file_point == NULL)
		printf("[-] <SaveCnn> Open write file failed <%s>\n", filename);

	for (int i = 0; i < cnn->C1->input_channels; i++)
		for (int j = 0; j < cnn->C1->output_channels; j++)
			for (int m = 0; m < cnn->C1->map_size; m++)
				fwrite(cnn->C1->map_data[i][j][m], sizeof(uint8_t),
					   cnn->C1->map_size, file_point);

	fwrite(cnn->C1->basic_data, sizeof(uint8_t),
		   cnn->C1->output_channels, file_point);

	for (int i = 0; i < cnn->C3->input_channels; i++)
		for (int j = 0; j < cnn->C3->output_channels; j++)
			for (int m = 0; m < cnn->C3->map_size; m++)
				fwrite(cnn->C3->map_data[i][j][m], sizeof(uint8_t),
					   cnn->C3->map_size, file_point);

	fwrite(cnn->C3->basic_data, sizeof(uint8_t),
		   cnn->C3->output_channels, file_point);

	for (int i = 0; i < cnn->O5->output_num; i++)
		fwrite(cnn->O5->wData[i], sizeof(uint8_t),
			   cnn->O5->input_num, file_point);

	fwrite(cnn->O5->basic_data, sizeof(uint8_t),
		   cnn->O5->output_num, file_point);

	for (int i = 0; i < cnn->O6->output_num; i++)
		fwrite(cnn->O6->wData[i], sizeof(uint8_t),
			   cnn->O6->input_num, file_point);

	fwrite(cnn->O6->basic_data, sizeof(uint8_t),
		   cnn->O6->output_num, file_point);

	fclose(file_point);
}

void CnnTrain(Cnn *cnn, ImageArray input_data, LabelArray output_data,
			  TrainOptions opts, int trainNum)
{
	cnn->L = (uint8_t *)malloc(trainNum * sizeof(uint8_t));
	int e;
	for (e = 0; e < opts.numepochs; e++)
	{
		printf("[+] --Training ... %d/%d\n", e, opts.numepochs);
		int n = 0;
		for (n = 0; n < trainNum; n++)
		{
			printf("[+] Training Process: %d / %d%c", n, trainNum, (char)13);

			CnnFF(cnn, input_data->image_point[n].image_data);
			CnnBP(cnn, output_data->label_point[n].LabelData);

#if SAVECNNDATA
			char *filedir = "output/cnn_data/";
			const char *filename = CombineStrings(filedir,
												  CombineStrings(IntToChar(n), ".cnn"));
			SaveCnnData(cnn, filename, input_data->image_point[n].ImgData);
#endif

			CnnApplyGrads(cnn, opts, input_data->image_point[n].image_data);

			CnnClear(cnn);
			uint8_t l = 0.0;
			int i;
			for (i = 0; i < cnn->O6->output_num; i++)
				l = l + cnn->e[i] * cnn->e[i];
			if (n == 0)
				cnn->L[n] = l / (uint8_t)2.0;
			else
				cnn->L[n] = cnn->L[n - 1] * 0.99 + 0.01 * l / (uint8_t)2.0;
		}
	}
}

void CnnFF(Cnn *cnn, uint8_t **input_data)
{
	// int output_sizeW = cnn->S2->input_width;
	// int output_sizeH = cnn->S2->input_height;

	MatSize map_size = {cnn->C1->map_size, cnn->C1->map_size};
	MatSize input_size = {cnn->C1->input_width, cnn->C1->input_height};
	MatSize output_size = {cnn->S2->input_width, cnn->S2->input_height};
	for (int i = 0; i < (cnn->C1->output_channels); i++)
	{
		for (int j = 0; j < (cnn->C1->input_channels); j++)
		{
			uint8_t **mapout = 0;
			if (cnn->C1->mode_conv == 0)
				mapout = MatCov(cnn->C1->map_data[j][i], map_size,
								input_data, input_size, FULL);
			else if (cnn->C3->mode_conv == 1)
				mapout = MatCov(cnn->C1->map_data[j][i], map_size,
								input_data, input_size, SAME);
			else
				mapout = MatCov(cnn->C1->map_data[j][i], map_size,
								input_data, input_size, VALID);

			MatAdd(cnn->C1->v[i], cnn->C1->v[i], output_size, mapout, output_size);

			for (int row = 0; row < output_size.rows; row++)
				free(mapout[row]);

			free(mapout);
		}

		for (int row = 0; row < output_size.rows; row++)
			for (int col = 0; col < output_size.columns; col++)
				cnn->C1->y[i][row][col] = ActivationReLu(cnn->C1->v[i][row][col],
														 cnn->C1->basic_data[i]);
	}

	output_size.columns = cnn->C3->input_width;
	output_size.rows = cnn->C3->input_height;

	input_size.columns = cnn->S2->input_width;
	input_size.rows = cnn->S2->input_height;

	for (int i = 0; i < (cnn->S2->output_channels); i++)
	{
		if (cnn->S2->pooling_type == AVG_POOLING)
			AvgPooling(cnn->S2->y[i], output_size, cnn->C1->y[i],
					   input_size, cnn->S2->map_size);
		else if (cnn->S2->pooling_type == MAX_POOLING)
			MaxPooling(cnn->S2->y[i], output_size, cnn->C1->y[i],
					   input_size, cnn->S2->map_size);
	}

	output_size.columns = cnn->S4->input_width;
	output_size.rows = cnn->S4->input_height;
	input_size.columns = cnn->C3->input_width;
	input_size.rows = cnn->C3->input_height;
	map_size.columns = cnn->C3->map_size;
	map_size.rows = cnn->C3->map_size;

	for (int i = 0; i < (cnn->C3->output_channels); i++)
	{
		for (int j = 0; j < (cnn->C3->input_channels); j++)
		{
			uint8_t **mapout = 0;
			if (cnn->C3->mode_conv == 0)
				mapout = MatCov(cnn->C3->map_data[j][i], map_size,
								cnn->S2->y[j], input_size, FULL);
			else if (cnn->C3->mode_conv == 1)
				mapout = MatCov(cnn->C3->map_data[j][i], map_size,
								cnn->S2->y[j], input_size, SAME);
			else
				mapout = MatCov(cnn->C3->map_data[j][i], map_size,
								cnn->S2->y[j], input_size, VALID);

			MatAdd(cnn->C3->v[i], cnn->C3->v[i], output_size, mapout, output_size);

			for (int r = 0; r < output_size.rows; r++)
				free(mapout[r]);

			free(mapout);
		}
		for (int r = 0; r < output_size.rows; r++)
			for (int c = 0; c < output_size.columns; c++)
				cnn->C3->y[i][r][c] = ActivationReLu(cnn->C3->v[i][r][c],
													 cnn->C3->basic_data[i]);
	}

	input_size.columns = cnn->S4->input_width;
	input_size.rows = cnn->S4->input_height;
	output_size.columns = input_size.columns / cnn->S4->map_size;
	output_size.rows = input_size.rows / cnn->S4->map_size;
	for (int i = 0; i < (cnn->S4->output_channels); i++)
	{
		if (cnn->S4->pooling_type == AVG_POOLING)
			AvgPooling(cnn->S4->y[i], output_size, cnn->C3->y[i],
					   input_size, cnn->S4->map_size);
		else if (cnn->S4->pooling_type == MAX_POOLING)
			MaxPooling(cnn->S4->y[i], output_size, cnn->C3->y[i],
					   input_size, cnn->S4->map_size);
	}

	uint8_t *O5inData = (uint8_t *)malloc((cnn->O5->input_num) * sizeof(uint8_t));

	for (int i = 0; i < (cnn->S4->output_channels); i++)
		for (int row = 0; row < output_size.rows; row++)
			for (int col = 0; col < output_size.columns; col++)
				O5inData[i * output_size.rows * output_size.columns +
						 row * output_size.columns + col] = cnn->S4->y[i][row][col];

	MatSize nnSize = {cnn->O5->input_num, cnn->O5->output_num};
	nnff(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->basic_data, nnSize);

	for (int i = 0; i < cnn->O5->output_num; i++)
		cnn->O5->y[i] = ActivationReLu(cnn->O5->v[i], cnn->O5->basic_data[i]);

	free(O5inData);

	uint8_t *O6inData = (uint8_t *)malloc((cnn->O6->input_num) * sizeof(uint8_t));

	for (int i = 0; i < (cnn->O5->output_num); i++)
		O6inData[i] = cnn->O5->y[i];

	nnSize.columns = cnn->O6->input_num;
	nnSize.rows = cnn->O6->output_num;

	nnff(cnn->O6->v, O6inData, cnn->O6->wData, cnn->O6->basic_data, nnSize);

	for (int i = 0; i < cnn->O6->output_num; i++)
		cnn->O6->y[i] = ActivationReLu(cnn->O6->v[i], cnn->O6->basic_data[i]);

	free(O6inData);
}

//
uint8_t ActivationSigma(uint8_t input, uint8_t bas) // sigma activatiion function
{
	uint8_t temp = input + bas;
	return (uint8_t)1.0 / ((uint8_t)(1.0 + exp(-temp)));
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
uint8_t ActivationReLu(uint8_t input, uint16_t bas)
{
	int8_t temp = 0;
	int8_t sum = 0;
	temp = input;
	sum = temp;
	return sum;
	// if ((sum > 32) && (sum <= 63))
	// {
	// 	return sum;
	// }
	// if (sum > 63)
	// 	return 63;
	// else if (sum <= 32)
	// {
	// 	return 32;
	// }
}
void AvgPooling(uint8_t **output, MatSize output_size, uint8_t **input,
				MatSize input_size, int map_size)
{
	int outputW = input_size.columns / map_size;
	int outputH = input_size.rows / map_size;

	if (output_size.columns != outputW || output_size.rows != outputH)
		printf("[-] ERROR: Output size is wrong! <AvgPooling> \n");

	int i, j, m, n;
	for (i = 0; i < outputH; i++)
		for (j = 0; j < outputW; j++)
		{
			uint8_t sum = 0.0;
			for (m = i * map_size; m < i * map_size + map_size; m++)
				for (n = j * map_size; n < j * map_size + map_size; n++)
					sum = sum + input[m][n];

			output[i][j] = sum / (uint8_t)(map_size * map_size);
		}
}

void MaxPooling(uint8_t ***output, MatSize output_size, uint8_t **input,
				MatSize input_size, int map_size)
/*it is just address storage */
{
	int outputW = input_size.columns / map_size;
	int outputH = input_size.rows / map_size;

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
//
uint8_t vecMulti(uint8_t *vec1, uint8_t *vec2, int vecL) //
{
	int i;
	uint8_t m = 0;
	for (i = 0; i < vecL; i++)
		m = m + vec1[i] * vec2[i];
	return m;
}

void nnff(uint8_t *output, uint8_t *input, uint8_t **wdata, uint8_t *bas, MatSize nnSize)
{
	int w = nnSize.columns;
	int h = nnSize.rows;

	int i;
	for (i = 0; i < h; i++)
		output[i] = vecMulti(input, wdata[i], w) + bas[i];
}

uint8_t sigma_derivation(uint8_t y)
{ //
	return y * (1 - y);
}

void CnnBP(Cnn *cnn, uint8_t *output_data)
{
	for (int i = 0; i < cnn->O6->output_num; i++)
		cnn->e[i] = cnn->O6->y[i] - output_data[i];

	for (int i = 0; i < cnn->O6->output_num; i++)
		// cnn->O6->d[i] = cnn->e[i] * sigma_derivation(cnn->O6->y[i]);
		cnn->O6->d[i] = cnn->e[i];

	MatSize output_size = {cnn->S4->input_width / cnn->S4->map_size,
						   cnn->S4->input_height / cnn->S4->map_size};

	for (int i = 0; i < cnn->S4->output_channels; i++)
		for (int row = 0; row < output_size.rows; row++)
			for (int col = 0; col < output_size.columns; col++)
				for (int j = 0; j < cnn->O5->output_num; j++)
				{
					int wInt = i * output_size.columns * output_size.rows +
							   row * output_size.columns + col;
					cnn->S4->d[i][row][col] = cnn->S4->d[i][row][col] +
											  cnn->O5->d[j] * cnn->O5->wData[j][wInt];
				}

	for (int i = 0; i < cnn->O5->output_num; i++)
		for (int j = 0; j < cnn->O6->output_num; j++)
		{
			int wInt = i;
			cnn->O5->d[i] = cnn->O5->d[i] +
							cnn->O6->d[j] * cnn->O6->wData[j][wInt];
		}

	int mapdata = cnn->S4->map_size;
	MatSize S4dSize;
	S4dSize.columns = cnn->S4->input_width / mapdata;
	S4dSize.rows = cnn->S4->input_height / mapdata;

	for (int i = 0; i < cnn->C3->output_channels; i++)
	{
		uint8_t **C3e = MatUpSample(cnn->S4->d[i], S4dSize,
									cnn->S4->map_size, cnn->S4->map_size);

		for (int row = 0; row < cnn->S4->input_height; row++)
			for (int col = 0; col < cnn->S4->input_width; col++)
			{
				// cnn->C3->d[i][row][col] = C3e[row][col] *
				//   sigma_derivation(cnn->C3->y[i][row][col]) /
				//   (uint8_t)(cnn->S4->map_size * cnn->S4->map_size);
				cnn->C3->d[i][row][col] = C3e[row][col] /
										  (uint8_t)(cnn->S4->map_size * cnn->S4->map_size);
			}

		for (int row = 0; row < cnn->S4->input_height; row++)
			free(C3e[row]);
		free(C3e);
	}

	output_size.columns = cnn->C3->input_width;
	output_size.rows = cnn->C3->input_height;
	MatSize input_size = {cnn->S4->input_width, cnn->S4->input_height};
	MatSize map_size = {cnn->C3->map_size, cnn->C3->map_size};

	for (int i = 0; i < cnn->S2->output_channels; i++)
	{
		for (int j = 0; j < cnn->C3->output_channels; j++)
		{
			uint8_t **corr = MatCorrelation(cnn->C3->map_data[i][j], map_size,
											cnn->C3->d[j], input_size, FULL);
			MatAdd(cnn->S2->d[i], cnn->S2->d[i], output_size, corr, output_size);
			for (int row = 0; row < output_size.rows; row++)
				free(corr[row]);
			free(corr);
		}
		/*
		for(r=0;r<cnn->C3->input_height;r++)
			for(c=0;c<cnn->C3->input_width;c++)
		*/
	}

	mapdata = cnn->S2->map_size;

	MatSize S2dSize;
	S2dSize.columns = cnn->S2->input_width / mapdata;
	S2dSize.rows = cnn->S2->input_height / mapdata;

	for (int i = 0; i < cnn->C1->output_channels; i++)
	{
		uint8_t **C1e = MatUpSample(cnn->S2->d[i], S2dSize,
									cnn->S2->map_size, cnn->S2->map_size);
		for (int row = 0; row < cnn->S2->input_height; row++)
			for (int col = 0; col < cnn->S2->input_width; col++)
			{
				// cnn->C1->d[i][row][col] = C1e[row][col] *
				//   sigma_derivation(cnn->C1->y[i][row][col]) /
				//   (uint8_t)(cnn->S2->map_size * cnn->S2->map_size);

				cnn->C1->d[i][row][col] = C1e[row][col] /
										  (uint8_t)(cnn->S2->map_size * cnn->S2->map_size);
			}
		for (int row = 0; row < cnn->S2->input_height; row++)
			free(C1e[row]);

		free(C1e);
	}
}

// Apply Grad
void CnnApplyGrads(Cnn *cnn, TrainOptions opts, uint8_t **input_data)
{
	MatSize dSize = {cnn->S2->input_height, cnn->S2->input_width};
	MatSize ySize = {cnn->C1->input_height, cnn->C1->input_width};
	MatSize map_size = {cnn->C1->map_size, cnn->C1->map_size};

	for (int i = 0; i < cnn->C1->output_channels; i++)
	{
		for (int j = 0; j < cnn->C1->input_channels; j++)
		{
			uint8_t **flipinput_data = MatRotate180(input_data, ySize);
			uint8_t **C1dk = 0;
			if (cnn->C1->mode_conv == 0)
				C1dk = MatCov(cnn->C1->d[i], dSize, flipinput_data, ySize, FULL);
			else if (cnn->C1->mode_conv == 1)
				C1dk = MatCov(cnn->C1->d[i], dSize, flipinput_data, ySize, SAME);
			else
				C1dk = MatCov(cnn->C1->d[i], dSize, flipinput_data, ySize, VALID);

			MatMultifactor(C1dk, C1dk, map_size, -1 * opts.alpha);
			MatAdd(cnn->C1->map_data[j][i], cnn->C1->map_data[j][i],
				   map_size, C1dk, map_size);
			for (int row = 0; row < (dSize.rows - (ySize.rows - 1)); row++)
				free(C1dk[row]);

			free(C1dk);

			for (int row = 0; row < ySize.rows; row++)
				free(flipinput_data[row]);

			free(flipinput_data);
		}
		cnn->C1->basic_data[i] = cnn->C1->basic_data[i] -
								 opts.alpha * MatSum(cnn->C1->d[i], dSize);
	}

	dSize.columns = cnn->S4->input_width;
	dSize.rows = cnn->S4->input_height;
	ySize.columns = cnn->C3->input_width;
	ySize.rows = cnn->C3->input_height;

	map_size.columns = cnn->C3->map_size;
	map_size.rows = cnn->C3->map_size;

	for (int i = 0; i < cnn->C3->output_channels; i++)
	{
		for (int j = 0; j < cnn->C3->input_channels; j++)
		{
			uint8_t **flipinput_data = MatRotate180(cnn->S2->y[j], ySize);
			uint8_t **C3dk = 0;
			if (cnn->C3->mode_conv == 0)
				C3dk = MatCov(cnn->C3->d[i], dSize, flipinput_data, ySize, FULL);
			else if (cnn->C3->mode_conv == 1)
				C3dk = MatCov(cnn->C3->d[i], dSize, flipinput_data, ySize, SAME);
			else
				C3dk = MatCov(cnn->C3->d[i], dSize, flipinput_data, ySize, VALID);

			MatMultifactor(C3dk, C3dk, map_size, -1.0 * opts.alpha);
			MatAdd(cnn->C3->map_data[j][i], cnn->C3->map_data[j][i],
				   map_size, C3dk, map_size);
			for (int row = 0; row < (dSize.rows - (ySize.rows - 1)); row++)
				free(C3dk[row]);

			free(C3dk);

			for (int row = 0; row < ySize.rows; row++)
				free(flipinput_data[row]);

			free(flipinput_data);
		}
		cnn->C3->basic_data[i] = cnn->C3->basic_data[i] -
								 opts.alpha * MatSum(cnn->C3->d[i], dSize);
	}

	uint8_t *O5inData = (uint8_t *)malloc((cnn->O5->input_num) * sizeof(uint8_t));

	MatSize output_size;
	output_size.columns = cnn->S4->input_width / cnn->S4->map_size;
	output_size.rows = cnn->S4->input_height / cnn->S4->map_size;

	for (int i = 0; i < (cnn->S4->output_channels); i++)
		for (int row = 0; row < output_size.rows; row++)
			for (int col = 0; col < output_size.columns; col++)
				O5inData[i * output_size.rows * output_size.columns +
						 row * output_size.columns + col] = cnn->S4->y[i][row][col];

	for (int j = 0; j < cnn->O5->output_num; j++)
	{
		for (int i = 0; i < cnn->O5->input_num; i++)
			cnn->O5->wData[j][i] = cnn->O5->wData[j][i] -
								   opts.alpha * cnn->O5->d[j] * O5inData[i];

		cnn->O5->basic_data[j] = cnn->O5->basic_data[j] -
								 opts.alpha * cnn->O5->d[j];
	}
	free(O5inData);

	uint8_t *O6inData = (uint8_t *)malloc((cnn->O6->input_num) * sizeof(uint8_t));

	for (int i = 0; i < (cnn->O5->output_num); i++)
		O6inData[i] = cnn->O5->y[i];

	for (int j = 0; j < cnn->O6->output_num; j++)
	{
		for (int i = 0; i < cnn->O6->input_num; i++)
			cnn->O6->wData[j][i] = cnn->O6->wData[j][i] -
								   opts.alpha * cnn->O6->d[j] * O6inData[i];

		cnn->O6->basic_data[j] = cnn->O6->basic_data[j] -
								 opts.alpha * cnn->O6->d[j];
	}
	free(O6inData);
}

void CnnClear(Cnn *cnn)
{
	for (int j = 0; j < cnn->C1->output_channels; j++)
	{
		for (int row = 0; row < cnn->S2->input_height; row++)
		{
			for (int col = 0; col < cnn->S2->input_width; col++)
			{
				cnn->C1->d[j][row][col] = (uint8_t)0.0;
				cnn->C1->v[j][row][col] = (uint8_t)0.0;
				cnn->C1->y[j][row][col] = (uint8_t)0.0;
			}
		}
	}

	for (int j = 0; j < cnn->S2->output_channels; j++)
	{
		for (int row = 0; row < cnn->C3->input_height; row++)
		{
			for (int col = 0; col < cnn->C3->input_width; col++)
			{
				cnn->S2->d[j][row][col] = (uint8_t)0.0;
				cnn->S2->y[j][row][col] = (uint8_t)0.0;
			}
		}
	}

	for (int j = 0; j < cnn->C3->output_channels; j++)
	{
		for (int row = 0; row < cnn->S4->input_height; row++)
		{
			for (int col = 0; col < cnn->S4->input_width; col++)
			{
				cnn->C3->d[j][row][col] = (uint8_t)0.0;
				cnn->C3->v[j][row][col] = (uint8_t)0.0;
				cnn->C3->y[j][row][col] = (uint8_t)0.0;
			}
		}
	}

	for (int j = 0; j < cnn->S4->output_channels; j++)
	{
		for (int row = 0; row < cnn->S4->input_height / cnn->S4->map_size; row++)
		{
			for (int col = 0; col < cnn->S4->input_width / cnn->S4->map_size; col++)
			{
				cnn->S4->d[j][row][col] = (uint8_t)0.0;
				cnn->S4->y[j][row][col] = (uint8_t)0.0;
			}
		}
	}

	for (int n = 0; n < cnn->O5->output_num; n++)
	{
		cnn->O5->d[n] = (uint8_t)0.0;
		cnn->O5->v[n] = (uint8_t)0.0;
		cnn->O5->y[n] = (uint8_t)0.0;
	}

	for (int n = 0; n < cnn->O6->output_num; n++)
	{
		cnn->O6->d[n] = (uint8_t)0.0;
		cnn->O6->v[n] = (uint8_t)0.0;
		cnn->O6->y[n] = (uint8_t)0.0;
	}
}

void SaveCnnData(Cnn *cnn, const char *filename, uint8_t **inputdata)
{
	FILE *file_point = NULL;
	file_point = fopen(filename, "wb");
	if (file_point == NULL)
		printf("[-] <SvaeCNNData> Open Write file failed! <%s>\n", filename);

	for (int i = 0; i < cnn->C1->input_height; i++)
		fwrite(inputdata[i], sizeof(uint8_t), cnn->C1->input_width, file_point);

	for (int i = 0; i < cnn->C1->input_channels; i++)
		for (int j = 0; j < cnn->C1->output_channels; j++)
			for (int s = 0; s < cnn->C1->map_size; s++)
				fwrite(cnn->C1->map_data[i][j][s], sizeof(uint8_t),
					   cnn->C1->map_size, file_point);

	fwrite(cnn->C1->basic_data, sizeof(uint8_t),
		   cnn->C1->output_channels, file_point);

	for (int j = 0; j < cnn->C1->output_channels; j++)
	{
		for (int row = 0; row < cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->v[j][row], sizeof(uint8_t),
				   cnn->S2->input_width, file_point);
		}

		for (int row = 0; row < cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->d[j][row], sizeof(uint8_t),
				   cnn->S2->input_width, file_point);
		}

		for (int row = 0; row < cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->y[j][row], sizeof(uint8_t),
				   cnn->S2->input_width, file_point);
		}
	}

	for (int j = 0; j < cnn->S2->output_channels; j++)
	{
		for (int row = 0; row < cnn->C3->input_height; row++)
		{
			fwrite(cnn->S2->d[j][row], sizeof(uint8_t),
				   cnn->C3->input_width, file_point);
		}
		for (int row = 0; row < cnn->C3->input_height; row++)
		{
			fwrite(cnn->S2->y[j][row], sizeof(uint8_t),
				   cnn->C3->input_width, file_point);
		}
	}

	for (int i = 0; i < cnn->C3->input_channels; i++)
		for (int j = 0; j < cnn->C3->output_channels; j++)
			for (int row = 0; row < cnn->C3->map_size; row++)
				fwrite(cnn->C3->map_data[i][j][row], sizeof(uint8_t),
					   cnn->C3->map_size, file_point);

	fwrite(cnn->C3->basic_data, sizeof(uint8_t),
		   cnn->C3->output_channels, file_point);

	for (int j = 0; j < cnn->C3->output_channels; j++)
	{
		for (int row = 0; row < cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->v[j][row], sizeof(uint8_t),
				   cnn->S4->input_width, file_point);
		}

		for (int row = 0; row < cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->d[j][row], sizeof(uint8_t),
				   cnn->S4->input_width, file_point);
		}

		for (int row = 0; row < cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->y[j][row], sizeof(uint8_t),
				   cnn->S4->input_width, file_point);
		}
	}

	for (int j = 0; j < cnn->S4->output_channels; j++)
	{
		for (int row = 0; row < cnn->S4->input_height / cnn->S4->map_size; row++)
		{
			fwrite(cnn->S4->d[j][row], sizeof(uint8_t),
				   cnn->S4->input_width / cnn->S4->map_size, file_point);
		}

		for (int row = 0; row < cnn->S4->input_height / cnn->S4->map_size; row++)
		{
			fwrite(cnn->S4->y[j][row], sizeof(uint8_t),
				   cnn->S4->input_width / cnn->S4->map_size, file_point);
		}
	}

	for (int i = 0; i < cnn->O5->output_num; i++)
		fwrite(cnn->O5->wData[i], sizeof(uint8_t), cnn->O5->input_num, file_point);

	fwrite(cnn->O5->basic_data, sizeof(uint8_t), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->v, sizeof(uint8_t), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->d, sizeof(uint8_t), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->y, sizeof(uint8_t), cnn->O5->output_num, file_point);

	for (int i = 0; i < cnn->O6->output_num; i++)
		fwrite(cnn->O6->wData[i], sizeof(uint8_t), cnn->O6->input_num, file_point);

	fwrite(cnn->O6->basic_data, sizeof(uint8_t), cnn->O6->output_num, file_point);
	fwrite(cnn->O6->v, sizeof(uint8_t), cnn->O6->output_num, file_point);
	fwrite(cnn->O6->d, sizeof(uint8_t), cnn->O6->output_num, file_point);
	fwrite(cnn->O6->y, sizeof(uint8_t), cnn->O6->output_num, file_point);

	fclose(file_point);
}