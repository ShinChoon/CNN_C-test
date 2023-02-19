
#ifndef MAT_H_
#define MAT_H_

#define FULL 0
#define SAME 1
#define VALID 2
#include <stdint.h>
typedef struct Mat2DSize
{
	int columns;
	int rows;
} MatSize;

uint8_t **MatRotate180(uint8_t **mat, MatSize mat_size);
void MatAdd(uint8_t **res, uint8_t **mat1, MatSize mat_size1, uint8_t **mat2, MatSize mat_size2);

uint8_t **MatCorrelation(uint8_t **map, MatSize map_size, uint8_t **inputData, MatSize inSize, int type);

uint8_t **MatCov(uint8_t **map, MatSize map_size, uint8_t **inputData, MatSize inSize, int type);

uint8_t **MatUpSample(uint8_t **mat, MatSize mat_size, int upc, int upr);

uint8_t **MatEdgeExpand(uint8_t **mat, MatSize mat_size, int addc, int addr);

uint8_t **MatEdgeShrink(uint8_t **mat, MatSize mat_size, int shrinkc, int shrinkr);

void MatSaving(uint8_t **mat, MatSize mat_size, const char *filename);

void MatMultifactor(uint8_t **res, uint8_t **mat, MatSize mat_size, uint8_t factor);

uint8_t MatSum(uint8_t **mat, MatSize mat_size);

char *CombineStrings(char *a, char *b);

char *IntToChar(int i);

#endif