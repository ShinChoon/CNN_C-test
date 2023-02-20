// Main function of CNN Train and Test.
//

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "basicTest.h"
#include "imagedata.h"
#include "weights_bias.h"

#define offset 0X3000100 //?
#define tx_status_adr 0x30001008
#define address_bus 0x30001004
#define info_bus 0x30001004

void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size)
{
    int map_size = 3;
    cnn->layer_num = 5; // layers = 5
    int pool_scale = 2;

    MatSize temp_input_size;

    // Layer1 Cov input size: {28,28}
    temp_input_size.columns = input_size.columns;
    temp_input_size.rows = input_size.rows;

    printf("temp_input_size col: %d\n", temp_input_size.columns);
    printf("temp_input_size rows: %d\n", temp_input_size.rows);

    cnn->C1 = InitialCovLayer(temp_input_size.columns,
                              temp_input_size.rows, map_size, 1, 4, VALID);

    // Layer2 Pooling input size: {28,28}
    temp_input_size.columns = temp_input_size.columns - map_size + 1;
    temp_input_size.rows = temp_input_size.rows - map_size + 1;
    printf("temp_input_size col: %d\n", temp_input_size.columns);
    printf("temp_input_size rows: %d\n", temp_input_size.rows);

    cnn->S2 = InitialPoolingLayer(temp_input_size.columns,
                                  temp_input_size.rows, pool_scale, 4, 4, MAX_POOLING);

    // // Layer3 Cov input size: {14,14}
    // temp_input_size.columns = temp_input_size.columns / 2;
    // temp_input_size.rows = temp_input_size.rows / 2;
    // printf("temp_input_size col: %d\n", temp_input_size.columns);
    // printf("temp_input_size rows: %d\n", temp_input_size.rows);

    // cnn->C3 = InitialCovLayer(temp_input_size.columns,
    //                           temp_input_size.rows, map_size, 4, 8, VALID);

    // // Layer4 Pooling with average. Input size: {12,12}
    // temp_input_size.columns = temp_input_size.columns - map_size + 1;
    // temp_input_size.rows = temp_input_size.rows - map_size + 1;
    // printf("temp_input_size col: %d\n", temp_input_size.columns);
    // printf("temp_input_size rows: %d\n", temp_input_size.rows);
    // cnn->S4 = InitialPoolingLayer(temp_input_size.columns,
    //                               temp_input_size.rows, pool_scale, 8, 8, MAX_POOLING);
}

void _CnnFF(CovLayer *conv_layer, PoolingLayer *pool_layer, uint8_t *bias_array)
/*
    1st Activation + Pooling
*/
{
    // MatSize map_size = {conv_layer->map_size, conv_layer->map_size};
    MatSize input_size = {conv_layer->input_width, conv_layer->input_height};
    MatSize output_size = {pool_layer->input_width, pool_layer->input_height};
    // int output_sizeW = pool_layer->input_width;
    // int output_sizeH = pool_layer->input_height;

    /*convolution result is conv_layer->v*/
    for (int i = 0; i < (conv_layer->output_channels); i++)
    {
        /*Activation function with params of weighted input and bias*/
        for (int row = 0; row < output_size.rows; row++)
        {
            for (int col = 0; col < output_size.columns; col++)
            {
                conv_layer->y[i][row][col] = ActivationReLu(conv_layer->v[i][row][col],
                                                            bias_array[i]);
            }
        }
    }
    output_size.columns = pool_layer->input_width / 2;
    output_size.rows = pool_layer->input_height / 2;
    input_size.columns = pool_layer->input_width;
    input_size.rows = pool_layer->input_height;
    for (int i = 0; i < (pool_layer->output_channels); i++)
    {
        if (pool_layer->pooling_type == AVG_POOLING)
            AvgPooling(pool_layer->y[i], output_size, conv_layer->y[i],
                       input_size, pool_layer->map_size);
        else if (pool_layer->pooling_type == MAX_POOLING)
            MaxPooling(pool_layer->y[i], output_size, conv_layer->y[i],
                       input_size, pool_layer->map_size);
    }
}

// Read one image from data <filename>
ImageArray _ReadImages(const char *filename)
/*try to add padding from 28x28 to 30x30, only return 1*/
{
    // Read images from file with file_point
    int number_of_images = 0; // Images' number
    int n_rows = 0;           // number of rows of an image<image hight>
    int n_columns = 0;        // number of cols of an image<image width>

    // uint8_t zero_pixel = 0;
    uint8_t temp_pixel = 0;

    number_of_images = 1;

    n_rows = 30;
    n_columns = 30;

    ImageArray image_array = malloc(sizeof(*image_array));
    // define strutrue of image array
    image_array->number_of_images = number_of_images; // number of images
    // array of all images.
    image_array->image_point = malloc(number_of_images * sizeof(*(image_array->image_point)));

    int row, column;                           // Temp for row and column
    for (int i = 0; i < number_of_images; ++i) // Images from 0 -> number_of_images-1
    {
        image_array->image_point[i].number_of_rows = n_rows;
        image_array->image_point[i].number_of_columns = n_columns; // set
        image_array->image_point[i].image_data = malloc((n_rows) * sizeof(*(image_array->image_point[i].image_data)));

        /*from 0 -> Nth rows*/
        for (row = 0; row < n_rows; ++row)
        {
            image_array->image_point[i].image_data[row] = malloc((n_columns) * sizeof((image_array->image_point[i].image_data[row]))); // expanding to 30
            for (column = 0; column < n_columns; ++column)                                                                             // from 0 -> n_columns
            {
                // read a pixel 0-255 with 8-bit
                temp_pixel = (uint8_t)myimagearray[row][column];
                // Change 8-bit pixel to uint8_t.
                image_array->image_point[i].image_data[row][column] = (uint8_t)temp_pixel;
            }
        }
    }

    return image_array;
}

uint8_t *bias_mapping(CovLayer *cc)
{
    printf("below is bias map\n");
    uint8_t *VMM_bias_map;
    VMM_bias_map = _alloc_1darray(cc->output_channels);
    for (int i = 0; i < cc->output_channels; i++)
    {
        VMM_bias_map[i] = bias_1[i];
    }
    return VMM_bias_map;
}

uint8_t ***weights_mapping(CovLayer *cc, int *weights_number, int scaling)
/*
    mapping weights into 32*36 matrix
    param cc: convLayer for current layer
    param weights_number: pointer to numer counting weights pattern duplication
    param scaling: scaling number for reduce the weights pattern size
    return weights_mapping [scaling][IMCcol][IMCrow]
*/
{
    printf("below is weights map\n");
    // printf("inputchannel: %d\n", cc->input_channels);
    // printf("outputchannel: %d\n", cc->output_channels);

    /*convert 2D map to 1D, memory acclocate space*/
    uint8_t ***VMM_weights_map = _alloc_3darray(scaling, IMCcol, IMCrow);

    int input_channels = cc->input_channels;
    int output_channels = cc->output_channels;
    int map_num = cc->map_size * cc->map_size;

    uint8_t map_array[output_channels][input_channels][map_num];
    int k2 = 0;
    int drift_x = 0;
    int drift_y = 0;
    /*weights map for columns by output channels*/
    /*convert matrix from output*input*3*3 into output*input*9 */
    for (int j = 0; j < output_channels; j++)
        for (int i = 0; i < cc->input_channels; i++)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    map_array[j][i][k2] = weights_map_1[j][i][x][y];
                    k2++;
                }
            }
            k2 = 0;
        }

    /*convert map array shape: cascade element from each input channel*/
    uint8_t mid_map_array[output_channels][input_channels * map_num];

    for (int j = 0; j < output_channels; j++)
    {
        int row_index = 0;
        for (int x = 0; x < 9; x++)
        {
            for (int i = 0; i < cc->input_channels; i++)
            {
                mid_map_array[j][row_index] = map_array[j][i][x];
                row_index++;
            }
        }
    }

    /*for mapping through the IMC matrix 32*36, with drift in x and y direction*/
    for (int r = 0; r < IMCcol / output_channels; r++)
    {
        for (int h = 0; h < output_channels; h++) // for each column
        {
            int row_index = 0;

            for (int i = 0; i < 9 * (input_channels / scaling); i++)
            {

                for (int i = 0; i < drift_y; i++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        VMM_weights_map[sch][h + drift_x][i] = 0;
                    }
                }

                for (int sch = 0; sch < scaling; sch++)
                {
                    // for (int in_c = 0; in_c < cc->input_channels / scaling; in_c++)
                    // {
                    //     i = i + in_c;
                    VMM_weights_map[sch][h + drift_x][i + drift_y] = mid_map_array[h][row_index];
                    row_index++;
                    // }
                }

                for (int i = 9 * (input_channels / scaling) + drift_y; i < IMCrow; i++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        VMM_weights_map[sch][h + drift_x][i] = 0;
                    }
                }
            }
        }
        drift_x += 1 * output_channels;
        drift_y += 3 * input_channels / scaling;
    }

    /*counting patten duplication times*/
    for (int i = 0; i < IMCcol; i++)
    {
        if (((i + 1) % output_channels == 0) && (i > 1))
            (*weights_number)++;
    }

    return VMM_weights_map;
}

void inputs_mapping(CovLayer *cc, MnistImage **images, uint8_t ***maplist, int *VMM_turns, int scaling, int part_index)
/*Create 9x1 lines of image data and concatenate lines into 2D array*/
/*
    param images: image list
    param maplist: map list, 1x28x36 list, output
    param VMM_turns: number of VMM pages, output
    param scaling: scaling number
    param part_index: part_index of input mapping
*/
{
    MatSize temp_input_size;
    temp_input_size.columns = 30;
    temp_input_size.rows = 30;

    // printf("below is image map\n");

    /*convert 2D array to 1D*/
    uint8_t VMM_input[IMCrow];
    int size_xx = sizeof(VMM_input) / sizeof(VMM_input[0]);
    int count_x = 0;
    int count_y = 0;
    int index_VMM_input = 1;
    int columns_number = images[0]->number_of_columns;
    uint8_t temp_input[size_xx];
    uint8_t r = 0;
    uint8_t c = 0;

    /*could be for test only when there is no data feeding*/
    // cc->input_channels = 2;
    // cc->output_channels = 8;
    // temp_input_size.columns = 14;
    // temp_input_size.rows = 14;
    int base_index_x = IMCcol / (cc->input_channels / scaling * cc->output_channels) + 1;
    int base_index_y = IMCcol;
    columns_number = temp_input_size.rows;

    printf("scaling:%d\n", scaling);

    for (int scal = 0; scal < scaling; scal++)
        for (int d = 0; d < temp_input_size.rows - 3 + 1; d++) // base number for index in y direction(0:31)
        {
            for (int i = 0; i < base_index_x + cc->input_channels; i++) // base number for index in x direction from 0 to 9 in one page (0:1:9)
            {
                for (r = 0 + i * 3 * 1 / scaling;                                                                                         // initial state
                     (r < 3 * cc->input_channels / scaling + i * 3 * cc->input_channels / scaling) && (r < temp_input_size.columns); r++) // index by x direction in one VMM page (0:1:11):(8,8,8):(16:1:27)
                {
                    for (c = 0 + d;                                      // initial state
                         (c < d + 3) && (c < temp_input_size.rows); c++) // index by y direction in one vmm page[(0, 1, 2):[3, 3, 3]:(25, 26, 27)]
                    {
                        /*collect image data into input array*/
                        for (int ch = 0; ch < cc->input_channels / scaling; ch++)
                        {
                            VMM_input[count_x] = images[ch]->image_data[r][c];
                            count_x++;
                        }

                        // printf("index_VMM_input: %d\n", index_VMM_input);
                        index_VMM_input = index_VMM_input + cc->input_channels / scaling;
                        // printf("index_VMM_input: %d\n", index_VMM_input);

                        if (index_VMM_input > IMCrow)
                        {
                            index_VMM_input = 1;
                            if (r < columns_number - 1)
                            {
                                r -= 4 / (cc->input_channels / scaling);
                                count_x = 0;
                                assign_to_sub_array(maplist, VMM_input, size_xx, count_y, scal);

                                count_y++;
                            }
                            else
                            {
                                /*once it moved to the end, new VMM created with only 4 rows mapped*/
                                // printf("!!!\n");
                                count_x = 0;
                                /*store the current input array into list*/
                                assign_to_sub_array(maplist, VMM_input, size_xx, count_y, scal);
                                for (int h = 0; h < size_xx; h++)
                                {
                                    temp_input[h] = 0;
                                }

                                /*duplicate four lines from previous line*/
                                for (int h = 0; h < size_xx; h++)
                                    if ((h < 4))
                                        temp_input[h] = VMM_input[size_xx + h - 4];

                                /*store the new input array into list*/
                                count_y++;
                                assign_to_sub_array(maplist, temp_input, size_xx, count_y, scal);
                                /*go to next line*/
                                count_y++;
                            }
                            if ((count_y % ((part_index + 1) * 28) == 0) && (count_y > (part_index * 28)))
                            {
                                *VMM_turns = count_y;
                                /*use this  as a range swtich!*/
                                return 0;
                            }
                        }
                    }
                }
            }
            if ((r >= temp_input_size.columns) && (c >= temp_input_size.columns))
                break;
        }
    // /*debug*/
    // printf("\ncount_x: %d\n", count_x);
    // printf("\ncount_y: %d\n", count_y);
    // printf("input size: %d\n", temp_input_size.columns);
    // printf("base_index_x: %d\n", base_index_x);

    // printf("input channel: %d\n", 1 / scaling);
    // printf("output channel: %d\n", 4);

    *VMM_turns = count_y;
}

const char *getfield(char *line, int num)
{
    const char *tok;
    for (tok = strtok(line, " []\n\r");
         tok && *tok;
         tok = strtok(NULL, " []\n\r"))
    {
        if (!--num)
            return tok;
    }
    return tok;
}

void _ImportCnn(Cnn *cnn)
// import cnn from header file
{
    for (int i = 0; i < cnn->C1->output_channels; i++)
        for (int j = 0; j < cnn->C1->input_channels; j++)
        {
            for (int r = 0; r < cnn->C1->map_size; r++)
            {
                for (int c = 0; c < cnn->C1->map_size; c++)
                {
                    cnn->C1->map_data[i][j][r][c] = weights_map_1[i][j][r][c];
                }
            }
        }

    for (int ch = 0; ch < cnn->C1->output_channels; ch++)
    {
        cnn->C1->basic_data[ch] = bias_1[ch];
    }
}

void assign_to_sub_array(uint8_t ***maplist, uint8_t *temp_input, int size_xx, int count_y, int scal)
{
    int sub_array_index = count_y / 28;
    for (int h = 0; h < size_xx; h++)
        maplist[scal][count_y - sub_array_index * 28][h] = temp_input[h];
}

void load_weights(CovLayer *cc, uint8_t ****weights)
{
    for (int i = 0; i < cc->output_channels; i++)
        for (int j = 0; j < cc->input_channels; j++)
        {
            for (int r = 0; r < cc->map_size; r++)
            {
                for (int c = 0; c < cc->map_size; c++)
                {
                    cc->map_data[i][j][r][c] = weights[i][j][r][c];
                }
            }
        }
}

void load_bias(CovLayer *cc, uint8_t *bias)
{
    for (int ch = 0; ch < cc->output_channels; ch++)
    {
        cc->basic_data[ch] = bias[ch];
    }
}

MnistImage *Output_image(int cols, int rows, uint8_t **imagedata)
/*convert output into MnistImage structure*/
{
    MnistImage *imagemodel = malloc(sizeof(*imagemodel));
    imagemodel->image_data = imagedata;
    imagemodel->number_of_columns = cols;
    imagemodel->number_of_rows = rows;
    return imagemodel;
}

VMM *initializeVMM(Cnn *cnn)
/*initalize the VMM and return VMM*/
{
    VMM *vmm = malloc(sizeof(*vmm));
    vmm->Cnn = cnn;
    vmm->cols = IMCcol;
    vmm->rows = IMCrow;
    vmm->MACoperation = MACoperation;
    return vmm;
}

uint8_t ***MACoperation(uint8_t ***input_array, uint8_t ***weight_array, int VMM_turns, int scaling)
{
    /*input array [78][36] weight_array[32][36]*/
    uint8_t ***output_array = _alloc_3darray(scaling, VMM_turns, IMCcol);
    for (int sc = 0; sc < scaling; sc++)
    {
        for (int i = 0; i < VMM_turns; i++)
        {
            // printf("calculate %d \n", i);
            for (int h = 0; h < IMCcol; h++)
            /*loop for 32 times in each column*/
            {
                /*parallel MAC will end here (I hope so!)*/
                for (int d = 0; d < IMCrow; d++)
                /*loop for 36 times in each row*/
                {
                    output_array[sc][i][h] += input_array[sc][i][d] * weight_array[sc][h][d];
                }
            }
        }
    }

    // /*create file for test*/
    // FILE *fpt;
    // fpt = fopen("MAC_array_0.csv", "w+");
    // for (int i = 0; i < VMM_turns; i++)
    // {
    //     for (int r = 0; r < IMCcol; r++)
    //     {
    //         fprintf(fpt, "%f ", output_array[i][r]);
    //     }
    //     fprintf(fpt, "\n");
    // }

    // fclose(fpt);

    // /*debug*/
    // for (int d = 0; d < VMM_turns; d++)
    // {
    //     for (int r = 0; r < IMCcol; r++)
    //         printf("%f  ", output_array[d][r]);
    //     printf("\n");
    // }

    return output_array;
}

void Conv_image(CovLayer *conv_layer, int input_width, uint8_t ***input_array,
                int VMM_turns, int weights_number, int scaling, int *column_index)
{
    int channels_number = conv_layer->output_channels;
    /*in each VMM turns*/
    int row_index = 0;
    int leftover_number = input_width % weights_number;
    int page_index = VMM_turns / 28;
    uint8_t mac_in_process = 0;
    uint8_t mac_in_end = 0;
    int d = 0;
    int scl = 0;
    int count = 0;

    for (int i = VMM_turns - 28; i < VMM_turns; i++)
    {
        if (((i + 1) % channels_number == 0) && (i > 1))
        /* when it comes to end of coulmns*/
        {
            for (int h = 0; h < leftover_number * channels_number; h++)
            {
                /*for each scanning x 4*/
                if (((h + 1) % channels_number == 0) && (h > 1))
                {
                    for (d = 0; d < channels_number; d++)
                    {
                        /*assign value from i VMM turn for dth channel, ith column, h element*/
                        for (scl = 0; scl < scaling; scl++)
                        {
                            mac_in_end += input_array[scl][i - 28 * (page_index - 1)][h + d - 3];
                        }
                        conv_layer->v[d][row_index][*column_index] = mac_in_end;
                        // printf("mac_in_end: %d\n", (int)(mac_in_end*1000));
                        mac_in_end = 0;
                    }
                    row_index++;
                    // printf("finsihed\n");
                }
            }

            row_index = 0;
            (*column_index)++;
            // if ((*column_index) % 28 == 0)
            // printf("\n");
        }
        else
        /*when it is on the way*/
        {
            for (int h = 0; h < IMCcol; h++)
            {
                /*for each scanning x 4*/
                if (((h + 1) % channels_number == 0) && (h > 1))
                {
                    for (d = 0; d < channels_number; d++)
                    {
                        /*assign value from i VMM turn for dth channel, ith column, h element*/
                        for (scl = 0; scl < scaling; scl++)
                        {
                            mac_in_process += input_array[scl][i - 28 * (page_index - 1)][h + d - 3];
                        }
                        conv_layer->v[d][row_index][*column_index] = mac_in_process;
                        // printf("mac_in_process: %d\n", (int)(mac_in_process * 1000));
                        mac_in_process = 0;
                    }
                    row_index++;
                    // printf("finished\n");
                }
            }
        }
    }
}

void save_image(int scale, uint8_t **image_data)
{
    int temp = 0;
    for (int i = 0; i < scale; i++)
    {
        for (int j = 0; j < scale; j++)
        {
            temp = (int)(image_data[i][j]);
            // Writing the gray values in the 2D array to the file
            printf("%d ", temp);
        }
        printf("\n");
    }
}

void freeConvLayer(CovLayer *covL)
/*free space of Convolutional layer*/
{
    printf("freeConvLayer!\n");
    int i, j, c, r;
    for (i = 0; i < covL->output_channels; i++)
    {
        for (j = 0; j < covL->input_channels; j++)
        {
            for (r = 0; r < covL->map_size; r++)
            {
                free(covL->map_data[i][j][r]);
            }
            free(covL->map_data[i][j]);
        }
        free(covL->map_data[i]);
    }
    free(covL->map_data);
    printf("free map_data!\n");
    free(covL->basic_data);
    printf("free basic_data!\n");

    int outW = covL->input_width - covL->map_size + 1;
    int outH = covL->input_height - covL->map_size + 1;

    for (j = 0; j < covL->output_channels; j++)
    {
        for (r = 0; r < outH; r++)
        {
            free(covL->v[j][r]);
            free(covL->y[j][r]);
        }
        free(covL->v[j]);
        free(covL->y[j]);
    }
    free(covL->v);
    free(covL->y);
    free(covL);
}

void freePoolLayer(PoolingLayer *pol)
{

    int outW = pol->input_width / pol->map_size;
    int outH = pol->input_height / pol->map_size;

    int j, r;
    for (j = 0; j < pol->output_channels; j++)
    {
        for (r = 0; r < outH; r++)
        {
            free(pol->y[j][r]);
        }
        free(pol->y[j]);
    }
    free(pol->y);
    free(pol->basic_data);
    free(pol);
}

uint8_t ***generate_input_array()
{
    uint8_t ***VMM_input_array;
    VMM_input_array = malloc(sizeof(*VMM_input_array) * 1);
    VMM_input_array[0] = malloc(sizeof(*VMM_input_array[0]) * 28);
    for (int i = 0; i < 28; i++)
        VMM_input_array[0][i] = malloc(sizeof(*VMM_input_array[0][i]) * IMCrow);

    return VMM_input_array;
}

uint8_t ***generate_result_array()
{
    uint8_t ***result_list;
    result_list = malloc(sizeof(*result_list) * 1);
    result_list[0] = malloc(sizeof(*result_list[0]) * 28);
    for (int i = 0; i < 28; i++)
        result_list[0][i] = malloc(sizeof(*result_list[0][i]) * IMCcol);

    return result_list;
}

void _free_3darray(int ***data, size_t xlen, size_t ylen)
{
    size_t i, j;

    for (i = 0; i < xlen; ++i)
    {
        if (data[i] != NULL)
        {
            for (j = 0; j < ylen; ++j)
                free(data[i][j]);
            free(data[i]);
        }
    }
    free(data);
}

void _free_2darray(int **data, size_t xlen)
{
    size_t i, j;

    for (i = 0; i < xlen; ++i)
    {
        if (data[i] != NULL)
            free(data[i]);
    }
    free(data);
}

void _free_1darray(int *data)
{
    free(data);
}

uint8_t ***_alloc_3darray(size_t xlen, size_t ylen, size_t zlen)
{
    uint8_t ***p;
    size_t i, j;

    if ((p = malloc(xlen * sizeof *p)) == NULL)
    {
        perror("malloc 1");
        return NULL;
    }

    for (i = 0; i < xlen; ++i)
        p[i] = NULL;

    for (i = 0; i < xlen; ++i)
        if ((p[i] = malloc(ylen * sizeof *p[i])) == NULL)
        {
            perror("malloc 2");
            free_data(p, xlen, ylen);
            return NULL;
        }

    for (i = 0; i < xlen; ++i)
        for (j = 0; j < ylen; ++j)
            p[i][j] = NULL;

    for (i = 0; i < xlen; ++i)
        for (j = 0; j < ylen; ++j)
            if ((p[i][j] = malloc(zlen * sizeof *p[i][j])) == NULL)
            {
                perror("malloc 3");
                free_data(p, xlen, ylen);
                return NULL;
            }

    return p;
}

uint8_t **_alloc_2darray(size_t xlen, size_t ylen)
{
    uint8_t **p;
    size_t i, j;

    if ((p = malloc(xlen * sizeof *p)) == NULL)
    {
        perror("malloc 1");
        return NULL;
    }

    for (i = 0; i < xlen; ++i)
        p[i] = NULL;

    for (i = 0; i < xlen; ++i)
        if ((p[i] = malloc(ylen * sizeof *p[i])) == NULL)
        {
            perror("malloc 2");
            free_data(p, xlen, ylen);
            return NULL;
        }

    return p;
}

uint8_t *_alloc_1darray(size_t xlen)
{
    uint8_t *p;
    size_t i, j;

    if ((p = malloc(xlen * sizeof *p)) == NULL)
    {
        perror("malloc 1");
        return NULL;
    }

    return p;
}
