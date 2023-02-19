//
//

// #include <math.h>
// #include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "mat.h"

uint8_t **MatRotate180(uint8_t **mat, MatSize mat_size)
{
	int column, row;
	int outSizeW = mat_size.columns;
	int outSizeH = mat_size.rows;
	uint8_t **outputData = (uint8_t **)malloc(outSizeH * sizeof(uint8_t *));

	for (int i = 0; i < outSizeH; i++)
		outputData[i] = (uint8_t *)malloc(outSizeW * sizeof(uint8_t));

	for (row = 0; row < outSizeH; row++)
		for (column = 0; column < outSizeW; column++)
			outputData[row][column] = mat[outSizeH - row - 1][outSizeW - column - 1];

	return outputData;
}

uint8_t **MatCorrelation(uint8_t **map, MatSize map_size, uint8_t **inputData, MatSize inSize, int type)
{
	int halfmapsizew;
	int halfmapsizeh;
	if (map_size.rows % 2 == 0 && map_size.columns % 2 == 0)
	{
		halfmapsizew = (map_size.columns) / 2;
		halfmapsizeh = (map_size.rows) / 2;
	}
	else
	{
		halfmapsizew = (map_size.columns - 1) / 2;
		halfmapsizeh = (map_size.rows - 1) / 2;
	}

	//
	int outSizeW = inSize.columns + (map_size.columns - 1); //
	int outSizeH = inSize.rows + (map_size.rows - 1);
	uint8_t **outputData = (uint8_t **)malloc(outSizeH * sizeof(uint8_t *)); //
	for (int i = 0; i < outSizeH; i++)
		outputData[i] = (uint8_t *)calloc(outSizeW, sizeof(uint8_t));

	//
	uint8_t **exInputData = MatEdgeExpand(inputData, inSize, map_size.columns - 1, map_size.rows - 1);

	for (int j = 0; j < outSizeH; j++)
		for (int i = 0; i < outSizeW; i++)
			for (int r = 0; r < map_size.rows; r++)
				for (int c = 0; c < map_size.columns; c++)
				{
					outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
				}

	for (int i = 0; i < inSize.rows + 2 * (map_size.rows - 1); i++)
		free(exInputData[i]);
	free(exInputData);

	MatSize outSize = {outSizeW, outSizeH};
	switch (type)
	{
	case FULL:
		return outputData;
	case SAME:
	{
		uint8_t **sameres = MatEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh);
		for (int i = 0; i < outSize.rows; i++)
			free(outputData[i]);
		free(outputData);
		return sameres;
	}
	case VALID:
	{
		uint8_t **validres;
		if (map_size.rows % 2 == 0 && map_size.columns % 2 == 0)
			validres = MatEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
		else
			validres = MatEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);
		for (int i = 0; i < outSize.rows; i++)
			free(outputData[i]);
		free(outputData);
		return validres;
	}
	default:
		return outputData;
	}
}

uint8_t **MatCov(uint8_t **map, MatSize map_size, uint8_t **inputData, MatSize inSize, int type)
{
	/*Convolution array in 2D*/
	uint8_t **flipmap = MatRotate180(map, map_size);
	uint8_t **res = MatCorrelation(flipmap, map_size, inputData, inSize, type);
	int i;
	for (i = 0; i < map_size.rows; i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

uint8_t **MatUpSample(uint8_t **mat, MatSize mat_size, int upc, int upr)
{
	int i, j, m, n;
	int c = mat_size.columns;
	int r = mat_size.rows;
	uint8_t **res = (uint8_t **)malloc((r * upr) * sizeof(uint8_t *));
	for (i = 0; i < (r * upr); i++)
		res[i] = (uint8_t *)malloc((c * upc) * sizeof(uint8_t));

	for (j = 0; j < r * upr; j = j + upr)
	{
		for (i = 0; i < c * upc; i = i + upc)
			for (m = 0; m < upc; m++)
				res[j][i + m] = mat[j / upr][i / upc];

		for (n = 1; n < upr; n++)
			for (i = 0; i < c * upc; i++)
				res[j + n][i] = res[j][i];
	}
	return res;
}

uint8_t **MatEdgeExpand(uint8_t **mat, MatSize mat_size, int addc, int addr)
{
	int i, j;
	int c = mat_size.columns;
	int r = mat_size.rows;
	uint8_t **res = (uint8_t **)malloc((r + 2 * addr) * sizeof(uint8_t *));
	for (i = 0; i < (r + 2 * addr); i++)
		res[i] = (uint8_t *)malloc((c + 2 * addc) * sizeof(uint8_t));

	for (j = 0; j < r + 2 * addr; j++)
	{
		for (i = 0; i < c + 2 * addc; i++)
		{
			if (j < addr || i < addc || j >= (r + addr) || i >= (c + addc))
				res[j][i] = (uint8_t)0.0;
			else
				res[j][i] = mat[j - addr][i - addc];
		}
	}
	return res;
}

uint8_t **MatEdgeShrink(uint8_t **mat, MatSize mat_size, int shrinkc, int shrinkr)
{
	int i, j;
	int c = mat_size.columns;
	int r = mat_size.rows;
	uint8_t **res = (uint8_t **)malloc((r - 2 * shrinkr) * sizeof(uint8_t *));
	for (i = 0; i < (r - 2 * shrinkr); i++)
		res[i] = (uint8_t *)malloc((c - 2 * shrinkc) * sizeof(uint8_t));

	for (j = 0; j < r; j++)
	{
		for (i = 0; i < c; i++)
		{
			if (j >= shrinkr && i >= shrinkc && j < (r - shrinkr) && i < (c - shrinkc))
				res[j - shrinkr][i - shrinkc] = mat[j][i];
		}
	}
	return res;
}

void MatSaving(uint8_t **mat, MatSize mat_size, const char *filename)
{
	FILE *fp = NULL;
	fp = fopen(filename, "wb");
	if (fp == NULL)
		printf("write file failed\n");

	int i;
	for (i = 0; i < mat_size.rows; i++)
		fwrite(mat[i], sizeof(uint8_t), mat_size.columns, fp);
	fclose(fp);
}

void MatAdd(uint8_t **res, uint8_t **mat1, MatSize mat_size1, uint8_t **mat2, MatSize mat_size2)
{
	/*accumulation from 2D to 1D*/
	int i, j;
	if (mat_size1.columns != mat_size2.columns || mat_size1.rows != mat_size2.rows)
		printf("ERROR: Size is not same!");

	for (i = 0; i < mat_size1.rows; i++)
		for (j = 0; j < mat_size1.columns; j++)
			res[i][j] = mat1[i][j] + mat2[i][j];
}

void MatMultifactor(uint8_t **res, uint8_t **mat, MatSize mat_size, uint8_t factor)
{
	int i, j;
	for (i = 0; i < mat_size.rows; i++)
		for (j = 0; j < mat_size.columns; j++)
			res[i][j] = mat[i][j] * factor;
}

uint8_t MatSum(uint8_t **mat, MatSize mat_size)
{
	uint8_t sum = 0.0;
	int i, j;
	for (i = 0; i < mat_size.rows; i++)
		for (j = 0; j < mat_size.columns; j++)
			sum = sum + mat[i][j];
	return sum;
}