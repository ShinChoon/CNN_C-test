// Main function of CNN Train and Test.
//

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "basicTest.h"
#include "weights_bias.h"
#include "vmm.h"

#define offset 0X3000100 //?
#define tx_status_adr 0x30001008
#define address_bus 0x30001004
#define info_bus 0x30001004

void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size, int i)
{
    int map_size = 3;
    cnn->layer_num = 5; // layers = 5
    int pool_scale = 2;

    MatSize temp_input_size;
    if (i == 1)
    {
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
        // Layer3 Cov input size: {14,14}
        temp_input_size.columns = temp_input_size.columns / 2;
        temp_input_size.rows = temp_input_size.rows / 2;
        printf("temp_input_size col: %d\n", temp_input_size.columns);
        printf("temp_input_size rows: %d\n", temp_input_size.rows);
    }

    if (i == 2)
    {
        // Layer3 Cov input size: {14,14}
        temp_input_size.columns = input_size.columns;
        temp_input_size.rows = input_size.rows;

        printf("temp_input_size col: %d\n", temp_input_size.columns);
        printf("temp_input_size rows: %d\n", temp_input_size.rows);

        cnn->C3 = InitialCovLayer(temp_input_size.columns,
                                  temp_input_size.rows, map_size, 4, 8, VALID);

        // Layer4 Pooling with average. Input size: {12,12}
        temp_input_size.columns = temp_input_size.columns - map_size + 1;
        temp_input_size.rows = temp_input_size.rows - map_size + 1;
        printf("temp_input_size col: %d\n", temp_input_size.columns);
        printf("temp_input_size rows: %d\n", temp_input_size.rows);

        cnn->S4 = InitialPoolingLayer(temp_input_size.columns,
                                      temp_input_size.rows, pool_scale, 8, 8, MAX_POOLING);
    }
}

void _CnnFF(CovLayer *conv_layer, PoolingLayer *pool_layer)
/*
    1st Activation + Pooling
*/
{
    // MatSize map_size = {conv_layer->map_size, conv_layer->map_size};
    MatSize input_size = {conv_layer->input_width, conv_layer->input_height};
    MatSize output_size = {pool_layer->input_width, pool_layer->input_height};
    // int output_sizeW = pool_layer->input_width;
    // int output_sizeH = pool_layer->input_height;

    output_size.columns = pool_layer->input_width / 2;
    output_size.rows = pool_layer->input_height / 2;
    input_size.columns = pool_layer->input_width;
    input_size.rows = pool_layer->input_height;

    for (int i = 0; i < (pool_layer->output_channels); i++)
    {
        if (pool_layer->pooling_type == AVG_POOLING)
            AvgPooling(pool_layer->y[i], output_size, conv_layer->v[i],
                       input_size, pool_layer->map_size);
        else if (pool_layer->pooling_type == MAX_POOLING)
            MaxPooling(pool_layer->y[i], output_size, conv_layer->v[i],
                       input_size, pool_layer->map_size);
    }
}

// // Read one image from data <filename>
// ImageArray _ReadImages(const char *filename)
// /*try to add padding from 28x28 to 30x30, only return 1*/
// {
//     // Read images from file with file_point
//     int number_of_images = 0; // Images' number
//     int n_rows = 0;           // number of rows of an image<image hight>
//     int n_columns = 0;        // number of cols of an image<image width>

//     // uint8_t zero_pixel = 0;
//     uint8_t temp_pixel = 0;

//     number_of_images = 1;

//     n_rows = 30;
//     n_columns = 30;

//     ImageArray image_array = calloc(1, sizeof(*image_array));
//     // define strutrue of image array
//     image_array->number_of_images = number_of_images; // number of images
//     // array of all images.
//     image_array->image_point = calloc(number_of_images, sizeof(*(image_array->image_point)));

//     int row, column;                           // Temp for row and column
//     for (int i = 0; i < number_of_images; ++i) // Images from 0 -> number_of_images-1
//     {
//         image_array->image_point[i].number_of_rows = n_rows;
//         image_array->image_point[i].number_of_columns = n_columns; // set
//         image_array->image_point[i].image_data = calloc(n_rows, sizeof(*(image_array->image_point[i].image_data)));

//         /*from 0 -> Nth rows*/
//         for (row = 0; row < n_rows; ++row)
//         {
//             image_array->image_point[i].image_data[row] = calloc((n_columns), sizeof((image_array->image_point[i].image_data[row]))); // expanding to 30
//             for (column = 0; column < n_columns; ++column)                                                                            // from 0 -> n_columns
//             {
//                 // read a pixel 0-255 with 8-bit
//                 temp_pixel = (uint8_t)myimagearray[row][column];
//                 // Change 8-bit pixel to uint8_t.
//                 image_array->image_point[i].image_data[row][column] = (uint8_t)temp_pixel;
//             }
//         }
//     }

//     return image_array;
// }

void weights_mapping(CovLayer *cc, uint8_t ***VMM_weights_map, int *weights_number,
                     int scaling, int layer_index)
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

    int input_channels = cc->input_channels;
    int output_channels = cc->output_channels;
    int map_num = cc->map_size * cc->map_size;

    uint8_t map_array[output_channels][input_channels][map_num];
    uint8_t VMM_weights_map_test[scaling][IMCcol][IMCrow / 2];
    int k2 = 0;
    int drift_x = 0;
    int drift_y = 0;
    int _index_x_test_0 = 0;
    int _index_y_test_0 = 0;

    int _index_x_test_1 = 0;
    int _index_y_test_1 = 0;

    /*weights map for columns by output channels*/
    /*convert matrix from output*input*3*3 into output*input*9 */
    for (int j = 0; j < output_channels; j++)
        for (int i = 0; i < cc->input_channels; i++)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    if (layer_index == 1)
                        map_array[j][i][k2] = weights_map_1[j][i][x][y];
                    else
                        map_array[j][i][k2] = weights_map_2[j][i][x][y];
                    k2++;
                }
            }
            k2 = 0;
        }

    /*convert map array shape: cascade element from each input channel, as shape from 4x9 to 1x36*/
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
            int _row_index_test = 0;
            _index_x_test_0 = 0;
            // printf("%d ", mid_map_array[h][row_index]);

            for (int i = 0; i < 9 * (input_channels / scaling); i++)
            {

                for (int i = 0; i < drift_y; i++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        VMM_weights_map[sch][h + drift_x][i] = 0;
                    }
                }
            }

            if (layer_index > 1)
            {

                for (int i = 0; i < 9 * (input_channels / scaling); i++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        // for (int in_c = 0; in_c < cc->input_channels / scaling; in_c++)
                        {
                            if ((_row_index_test / 2) % 2 == 0)
                            {
                                // printf("odd ");
                                // printf("%d,%d->%d ", h + drift_x, _index_x_test_0-_row_index_test / 2, mid_map_array[h][_index_x_test_0]);
                                VMM_weights_map[0][h + drift_x][_index_x_test_0 - _row_index_test / 2 + drift_y] = mid_map_array[h][_index_x_test_0];
                                // VMM_weights_map_test[0][h][_index_x_test_0] = mid_map_array[h + drift_x][i + drift_y];
                            }
                            else
                            {
                                // printf("%d,%d->%d ", h + drift_x, _index_x_test_0 - _row_index_test / 2-1, mid_map_array[h][_index_x_test_0]);
                                VMM_weights_map[1][h + drift_x][_index_x_test_0 - _row_index_test / 2 - 1 + drift_y] = mid_map_array[h][_index_x_test_0];
                            }
                            // else
                            // printf("even ");
                            _index_x_test_0++;

                            _row_index_test++;
                            //             // i = i + in_c;
                            //             // printf("%d,%d->%d ", h + drift_x, i + drift_y, _index_x_test_0);
                            //             // printf(": %d   ", mid_map_array[h][row_index]);
                            //             // if ((row_index / 2) % 2 == 0)
                            //             // printf("odd ");
                            //             // else
                            //             // printf("even ");
                            //                 // // _index_x_test_0++;

                            //             // printf(": %d   ", mid_map_array[h][row_index]);
                            //             // _row_index_test++;
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < 9 * (input_channels / scaling); i++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        // for (int in_c = 0; in_c < cc->input_channels / scaling; in_c++)
                        {
                            // i = i + in_c;
                            VMM_weights_map[sch][h + drift_x][i + drift_y] = mid_map_array[h][row_index];
                            row_index++;
                        }
                    }
                }
            }

            for (int i = 0; i < 9 * (input_channels / scaling); i++)
            {

                for (int i = 9 * (input_channels / scaling) + drift_y; i < IMCrow; i++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        VMM_weights_map[sch][h + drift_x][i] = 0;
                    }
                }
            }
            // printf("\n");
        }

        drift_x += 1 * output_channels;
        drift_y += 3 * input_channels / scaling;
    }

    for (int i = 0; i < scaling; i++)
    {
        for (int j = 0; j < IMCcol; j++)
        {
            for (int z = 0; z < IMCrow; z++)
                printf("%d ", VMM_weights_map[i][j][z]);
            printf("\n");
        }
        printf("\n");
    }

    /*counting patten duplication times*/
    for (int i = 0; i < IMCcol; i++)
    {
        if (((i + 1) % output_channels == 0) && (i > 1))
            (*weights_number)++;
    }
}

void inputs_mapping(CovLayer *cc, uint8_t ***images, uint8_t ***maplist, int *VMM_turns,
                    int scaling, int layer_index)
/*Create 9x1 lines of image data and concatenate lines into 2D array*/
/*
    param images: image list
    param maplist: map list, 1x28x36 list, output
    param VMM_turns: number of VMM pages, output
    param scaling: scaling number
    param part_index: part_index of input mapping
    **please notice that the inputs mapping will have half space empty at each 4th([3]) page
*/
{
    MatSize temp_input_size;
    temp_input_size.columns = cc->input_width;
    temp_input_size.rows = cc->input_height;
    printf("temp_input_size.column:%d\n", temp_input_size.columns);
    printf("temp_input_size.rows:%d\n", temp_input_size.rows);

    /*convert 2D array to 1D*/
    uint8_t VMM_input[IMCrow]={0};
    int size_xx = IMCrow;
    uint8_t _local_VMM_input_lists[scaling][200][IMCrow];
    int count_x = 0;
    int count_y = 0;
    int index_VMM_input = 1;
    int columns_number = cc->input_height;
    uint8_t r = 0;
    uint8_t c = 0;
    uint8_t temp_input[size_xx];
    bool touched_end = false;

    /*could be for test only when there is no data feeding*/
    // cc->input_channels = 2;
    // cc->output_channels = 8;
    // temp_input_size.columns = 14;
    // temp_input_size.rows = 14;
    int base_index_x = IMCcol / (cc->input_channels / scaling * cc->output_channels) + 1;
    int base_index_y = IMCcol;
    columns_number = temp_input_size.rows;
    int back_step = layer_index==1? 6:0; // distance between the top of VMM to the highest kernel



    printf("scaling:%d\n", scaling);
    printf("base_index_x:%d\n", base_index_x);

    for (int scal = 0; scal < scaling; scal++)
    {

        for (int d = 0; d < temp_input_size.rows - 3 + 1; d++) // base number for index in y direction(0:31)
        {
            for (int i = 0; i < base_index_x + 1; i++) // base number for index in x direction from 0 to 9 in one page (0:1:9), here is a magic number 1 for unknow reasons
            {
                for (r = 0 + i * 3 * cc->input_channels / scaling;                                                                        // initial state
                     (r < 3 * cc->input_channels / scaling + i * 3 * cc->input_channels / scaling) && (r < temp_input_size.columns); r++) // index by x direction in one VMM page (0:1:11):(8,8,8):(16:1:27)
                {
                    for (c = 0 + d;                                      // initial state
                         (c < d + 3) && (c < temp_input_size.rows); c++) // index by y direction in one vmm page[(0, 1, 2):[3, 3, 3]:(25, 26, 27)]
                    {
                        /*collect image data into input array*/
                        for (int ch = 0; ch < cc->input_channels / scaling; ch++)
                        {
                            VMM_input[count_x] = images[ch + scaling * scal][r][c];// go through the image by channles and scaling
                            // if(VMM_input[count_x]!=0)
                            {
                                // printf("%d,%d  ", r, c);
                                // printf("%d  ", VMM_input[count_x]);
                            }
                            count_x++;// increment index in one column (36)
                        }
                        index_VMM_input +=  cc->input_channels / scaling;
                        if (r >= columns_number - 1 && index_VMM_input < IMCrow - back_step && layer_index==1)
                        {
                            index_VMM_input = IMCrow - back_step - 1;
                        }
                        if (index_VMM_input > IMCrow - back_step) // when input has been enough for one page
                        {
                            // printf("index_VMM_input: %d, r: %d ", index_VMM_input,r );
                            index_VMM_input = 1;
                            if (r < columns_number - 1) // if filter moves on the way
                            {
                                r -= 2 / (cc->input_channels / scaling); //duplicate four rows of the previous
                                // printf("@@\n");

                                for (int h = 0; h < size_xx; h++)
                                    _local_VMM_input_lists[scal][count_y][h] = VMM_input[h];

                                for (int i = 0; i < IMCrow; i++)
                                    VMM_input[i] = 0;

                                count_y++;
                            }

                            else// if filter touches the end of one column
                            {
                                /*once it moved to the end(new column coming next page), new VMM created with only 4 rows mapped*/
                                // printf("!!!\n");
                                /*store the current input array into list*/
                                for (int h = 0; h < size_xx; h++)
                                    _local_VMM_input_lists[scal][count_y][h] = VMM_input[h];

                                for (int i = 0; i < IMCrow; i++)
                                    VMM_input[i] = 0;

                                count_y++;
                            }
                            count_x = 0;//move count_x = 0; out of if condition logic
                        }

                    }
                }
            }
            if ((r >= temp_input_size.columns) && (c >= temp_input_size.columns))
                break;
        }
        printf("scal: %d\n", scal);
        printf("index_VMM_input: %d\n", index_VMM_input);
        printf("\n");
    }
    // /*debug*/
    printf("\ncount_x: %d\n", count_x);
    printf("\ncount_y: %d\n", count_y);
    // printf("input size: %d\n", temp_input_size.columns);
    // printf("base_index_x: %d\n", base_index_x);

    for (int d = 0; d < scaling; d++)
    {
        for (int i = 0; i < count_y; i++)
        {
            for (int h = 0; h < IMCrow; h++)
            {
                maplist[d][i][h] = _local_VMM_input_lists[d][i][h];
                // printf("%d ", maplist[d][i][h]);
            }
            // printf("\n");
        }
        // printf("\n");
    }

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

void _ImportCnn(Cnn *cnn, int i)
// import cnn from header file
{
    if (i == 1)
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
    else
    {
        for (int i = 0; i < cnn->C3->output_channels; i++)
            for (int j = 0; j < cnn->C3->input_channels; j++)
            {
                for (int r = 0; r < cnn->C3->map_size; r++)
                {
                    for (int c = 0; c < cnn->C3->map_size; c++)
                    {
                        cnn->C3->map_data[i][j][r][c] = weights_map_2[i][j][r][c];
                    }
                }
            }

        for (int ch = 0; ch < cnn->C3->output_channels; ch++)
        {
            cnn->C3->basic_data[ch] = bias_2[ch];
        }
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

ImageArray Output_image(int cols, int rows, uint8_t ***image_data, int number)
/*convert output into MnistImage structure*/
{
    // Read images from file with file_point

    int number_of_images = number; // Images' number
    int n_rows = rows;             // number of rows of an image<image hight>
    int n_columns = cols;          // number of cols of an image<image width>

    // uint8_t zero_pixel = 0;
    uint8_t temp_pixel = 0;
    ImageArray image_array = calloc(1, sizeof(*image_array));

    // define strutrue of image array
    image_array->number_of_images = number_of_images; // number of images
    // array of all images.
    image_array->image_point = calloc(number_of_images, sizeof(*(image_array->image_point)));

    for (int i = 0; i < number_of_images; ++i) // Images from 0 -> number_of_images-1
    {
        image_array->image_point[i].number_of_rows = n_rows;
        image_array->image_point[i].number_of_columns = n_columns; // set
        image_array->image_point[i].image_data = calloc((n_rows), sizeof(*(image_array->image_point[i].image_data)));

        /*from 0 -> Nth rows*/
        for (int row = 0; row < n_rows; ++row)
        {
            image_array->image_point[i].image_data[row] = calloc(n_columns, sizeof((image_array->image_point[i].image_data[row]))); // expanding to 30
            for (int column = 0; column < n_columns; ++column)                                                                      // from 0 -> n_columns
            {
                // read a pixel 0-255 with 8-bit
                /*if needed, replace image_data with debug image data*/
                temp_pixel = (uint8_t)image_data[i][row][column];
                // Change 8-bit pixel to uint8_t.
                image_array->image_point[i].image_data[row][column] = (uint8_t)temp_pixel;
            }
        }
    }

    return image_array;
}

VMM *initializeVMM(Cnn *cnn)
/*initalize the VMM and return VMM*/
{
    VMM *vmm = calloc(1, sizeof(*vmm));
    vmm->Cnn = cnn;
    vmm->cols = IMCcol;
    vmm->rows = IMCrow;
    vmm->MACoperation = &MACoperation;
    return vmm;
}

void MACoperation(CovLayer *conv_layer, uint8_t ***input_array, uint8_t ***output_array, uint8_t ***weight_array,
                  int page_image, int scaling)
            /*why the convolution layer output zero padding has only 4 rows???*/
{
    float fweight = 0;
    float fimage = 0;
    uint8_t uweight = 0;
    uint8_t uimage = 0;
    float dotproduct = 0;
    /*input array [78][36] weight_array[32][36]*/
    for (int sc = 0; sc < scaling; sc++)
    {
        {
            // printf("calculate %d \n", i);
            for (int h = 0; h < IMCcol; h++)
            /*loop for 32 times in each column*/
            {
                /*parallel MAC will end here (I hope so!)*/
                for (int d = 0; d < IMCrow; d++)
                /*loop for 36 times in each row*/
                {
                    fweight = bin_float_for_image_weights(weight_array[sc][h][d], 1);
                    fimage = bin_float_for_image_weights(input_array[sc][page_image][d], 0);
                    dotproduct += fweight * fimage;
                }
                if (conv_layer->output_channels == 4) // no need to process bias right??
                    dotproduct += bias_1[h % conv_layer->output_channels];
                else
                    dotproduct += bias_2[h % conv_layer->output_channels];

                // if (h % 4 == 0)
                // printf("%f ", dotproduct);
                output_array[sc][page_image][h] = float_bin_for_bias_result(dotproduct > 0 ? dotproduct : 0);
                dotproduct = 0;
            }
        }
    }

    // for (int sc = 0; sc < scaling; sc++)
    // {
        // // for (int i = 0; 0< i < VMM_turns; i++)
        // {
            // // printf("calculate %d \n", i);
            // // for (int h = 0; h < IMCcol; h++)
            // /*loop for 32 times in each column*/
            // {
                // /*parallel MAC will end here (I hope so!)*/
                // for (int d = 0; d < IMCrow; d++)
                // /*loop for 36 times in each row*/
                // {
                    // // fweight = bin_float_for_image_weights(weight_array[0][h][d], 1);
                    // fimage = bin_float_for_image_weights(input_array[0][page_image][d], 0);
                    // // if (fweight != 0)
                    // printf("%.2f ", fimage);
                // }
                // printf("\n");
            // }
        // }
    // }

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
    //         printf("%d  ", output_array[0][d][r]);
    //     printf("\n");
    // }
}

void Conv_image(CovLayer *conv_layer, PoolingLayer *pool_layer, uint8_t ***input_array,
                int VMM_turns, int weights_number, int scaling, int layer_index)
{
    int in_channel_number = conv_layer->input_channels;
    int out_channel_number = conv_layer->output_channels;
    /*in each VMM turns*/
    int row_index = 0;
    int column_index = 0;
    int leftover_number = pool_layer->input_width % weights_number;
    uint8_t mac_in_process = 0;
    uint8_t mac_in_end = 0;
    int d = 0;
    int count = 0;
    printf("pool_layer->input_width: %d\n", pool_layer->input_width);
    printf("leftover_number: %d\n", leftover_number);
    printf("weights_number: %d\n", weights_number);
    printf("remainder: %d\n", pool_layer->input_width % weights_number);
    int index = 0;
    for (int i = 0; i < VMM_turns; i++)
    {
        if (((i + 1) % (out_channel_number) == 0) && (i > 1)) // I really don't know what the fuck it is
        /* when it comes to end of coulmns*/
        {
            for (int h = 0; h < leftover_number * out_channel_number; h++)// here it defines that only half of resultlist will be taken
            {
                /*for each scanning x 4*/
                if (((h + 1) % out_channel_number == 0) && (h > 1))
                {
                    for (d = 0; d < out_channel_number; d++)
                    {
                        /*assign value from i VMM turn for dth channel, ith column, h element*/
                        for (int scl = 0; scl < scaling; scl++)
                        {
                            mac_in_end += input_array[scl][i][h + d - conv_layer->output_channels + 1];
                        }
                        // printf("leftover_number:%d, h:%d,  d:%d", leftover_number, h, d);

                        if (layer_index == 1)
                            conv_layer->v[d][row_index][column_index] = mac_in_end;
                        else
                        {
                            mac_in_end -= bias_2[d];
                            conv_layer->v[d][row_index][column_index] = mac_in_end;
                        }
                        // printf("mac_in_end: %d\n", (int)(mac_in_end*1000));
                        mac_in_end = 0;
                    }

                    if (row_index == conv_layer->input_width - conv_layer->map_size)
                    {
                        row_index = 0;
                    }
                    else
                        row_index++; // printf("finsihed\n");
                }
                // printf("\n");
            }

            if (leftover_number == 0) // for second layer case: leftover = 12%4 = 0, i+1 = 96 -> touch the end
            {
                for (int h = 0; h < IMCcol; h++)
                {
                    /*for each scanning x 4*/
                    if (((h + 1) % out_channel_number == 0) && (h > 1))
                    {
                        for (d = 0; d < out_channel_number; d++)
                        {
                            /*assign value from i VMM turn for dth channel, ith column, h element*/
                            for (int scl = 0; scl < scaling; scl++)
                            {
                                mac_in_process += input_array[scl][i][h + d - conv_layer->output_channels + 1];
                            }
                            // printf("%d  ", h + d - conv_layer->output_channels + 1);
                            // printf("d:%d, row_index: %d, *column_index: %d\n", row_index ,* column_index);
                            if (layer_index == 1)
                                conv_layer->v[d][row_index][column_index] = mac_in_process;
                            else
                                conv_layer->v[d][row_index][column_index] = mac_in_process;
                            // printf("mac_in_process: %d\n", (int)(mac_in_process * 1000));
                            mac_in_process = 0;
                        }
                        if (row_index == conv_layer->input_width - conv_layer->map_size)
                        {
                            row_index = 0;
                        }
                        else
                            row_index++; // printf("finished\n");
                    }
                }
            }
            // printf("\n");

            row_index = 0;
            column_index++;
            // if ((*column_index) % 28 == 0)
            // printf("\n");
        }
        else
        /*when it is on the way*/
        {
            for (int h = 0; h < IMCcol; h++)
            {
                /*for each scanning x 4*/
                if (((h + 1) % out_channel_number == 0) && (h > 1))
                {
                    for (d = 0; d < out_channel_number; d++)
                    {
                        /*assign value from i VMM turn for dth channel, ith column, h element*/
                        for (int scl = 0; scl < scaling; scl++)
                        {
                            mac_in_process += input_array[scl][i][h + d - conv_layer->output_channels + 1];
                        }
                        // printf("%d  ", h + d - conv_layer->output_channels + 1);
                        // printf("d:%d, row_index: %d, *column_index: %d\n", row_index ,* column_index);

                        if (layer_index == 1)
                            conv_layer->v[d][row_index][column_index] = mac_in_process;
                        else
                            conv_layer->v[d][row_index][column_index] = mac_in_process;

                        // printf("mac_in_process: %d\n", (int)(mac_in_process * 1000));
                        mac_in_process = 0;
                    }

                    if (row_index == conv_layer->input_width - conv_layer->map_size)
                    {
                        row_index = 0;
                    }
                    else
                        row_index++; // printf("finished\n");
                }
            }
            // printf("\n");
        }
    }
}

void save_image(int scale, uint8_t ***image_data)
{
    uint8_t temp = 0;
    for (int i = 0; i < scale; i++)
    {
        for (int j = 0; j < scale; j++)
        {
            temp = *(image_data[i][j]);
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
    int outW = covL->input_width - covL->map_size + 1;
    int outH = covL->input_height - covL->map_size + 1;

    for (j = 0; j < covL->output_channels; j++)
    {
        for (r = 0; r < outH; r++)
        {
            free(covL->v[j][r]);
            // free(covL->y[j][r]);
        }
        free(covL->v[j]);
        // free(covL->y[j]);
    }
    free(covL->v);
    // free(covL->y);
    free(covL);
}

void freePoolLayer(PoolingLayer *pol)
{
    printf("freePoolLayer!\n");
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
    // free(pol->basic_data);
    free(pol);
}

uint8_t ***generate_input_array(int scal, int size)
{
    uint8_t ***VMM_input_array = alloc_3darray(scal, size, IMCrow);

    return VMM_input_array;
}

uint8_t ***generate_result_array(int scal, int VMM_turns)
{
    uint8_t ***result_list = alloc_3darray(scal, VMM_turns, IMCcol);

    return result_list;
}

void free_data(uint8_t **data, size_t xlen, size_t ylen)
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

void free_3darray(uint8_t ***data, size_t xlen, size_t ylen)
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
    printf("done!\n");
}

void free_2darray(uint8_t **data, size_t xlen)
{
    size_t i, j;

    for (i = 0; i < xlen; ++i)
    {
        if (data[i] != NULL)
            free(data[i]);
    }
    free(data);
}

void free_image(ImageArray image_array, int rows, int number_of_images)
{
    for (int i = 0; i < number_of_images; ++i) // Images from 0 -> number_of_images-1
    {

        /*from 0 -> Nth rows*/
        for (int row = 0; row < rows; ++row)
        {
            free(image_array->image_point[i].image_data[row]);
        }
        free(image_array->image_point[i].image_data);
    }
    free(image_array->image_point);
    free(image_array);
}

uint8_t ***alloc_3darray(size_t xlen, size_t ylen, size_t zlen)
{
    uint8_t ***p;
    size_t i, j;

    if ((p = calloc(xlen, sizeof *p)) == NULL)
    {
        perror("calloc 1");
        return NULL;
    }

    for (i = 0; i < xlen; ++i)
        p[i] = NULL;

    for (i = 0; i < xlen; ++i)
        if ((p[i] = calloc(ylen, sizeof *p[i])) == NULL)
        {
            perror("calloc 2");
            free_3darray(p, xlen, ylen);
            return NULL;
        }

    for (i = 0; i < xlen; ++i)
        for (j = 0; j < ylen; ++j)
            p[i][j] = NULL;

    for (i = 0; i < xlen; ++i)
        for (j = 0; j < ylen; ++j)
            if ((p[i][j] = calloc(zlen, sizeof *p[i][j])) == NULL)
            {
                perror("calloc 3");
                free_3darray(p, xlen, ylen);
                return NULL;
            }

    return p;
}

uint8_t **alloc_2darray(size_t xlen, size_t ylen)
{
    uint8_t **p;
    size_t i, j;

    if ((p = calloc(xlen, sizeof *p)) == NULL)
    {
        perror("calloc 1");
        return NULL;
    }

    for (i = 0; i < xlen; ++i)
        p[i] = NULL;

    for (i = 0; i < xlen; ++i)
        if ((p[i] = calloc(ylen, sizeof *p[i])) == NULL)
        {
            perror("calloc 2");
            free_2darray(p, xlen);
            return NULL;
        }

    return p;
}

uint8_t *alloc_1darray(size_t xlen)
{
    uint8_t *p;
    size_t i, j;

    if ((p = calloc(xlen, sizeof *p)) == NULL)
    {
        perror("calloc 1");
        return NULL;
    }

    return p;
}
