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
/**
*@brief This function sets up the CNN network by initializing its different layers based on the given input_size and output_size parameters.
*
*@param cnn pointer to the CNN structure
*@param input_size size of the input matrix
*@param output_size number of output classes(not used right now)
*@param i index of the layer to be set up
*/
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

    if (i == 3)
    {
        temp_input_size.columns = input_size.columns;
        temp_input_size.rows = 1;
        cnn->O5 = InitOutputLayer(temp_input_size.columns * temp_input_size.rows,
                                  32);

        temp_input_size.columns = 32;
        temp_input_size.rows = 1;
        cnn->O6 = InitOutputLayer(temp_input_size.columns * temp_input_size.rows,
                                  10);
    }
}

/**
*@brief Performs forward pass of a convolutional layer followed by a pooling layer
*
*@param conv_layer Pointer to a ConvLayer structure representing the convolutional layer
*@param pool_layer Pointer to a PoolingLayer structure representing the pooling layer
*/

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
        // if (pool_layer->pooling_type == AVG_POOLING)
        //     AvgPooling(pool_layer->y[i], output_size, conv_layer->v[i],
        //                input_size, pool_layer->map_size);
        if (pool_layer->pooling_type == MAX_POOLING)
            MaxPooling(pool_layer->y[i], output_size, conv_layer->v[i],
                       input_size, pool_layer->map_size);
    }
}

/**
 * @brief Maps the convolution layer weights into a 32*36 matrix
 *
 * @param cc: ConvLayer for the current layer
 * @param VMM_weights_map: Pointer to the output 3D array [scaling][IMCcol][IMCrow]
 * @param weights_number: Pointer to the number counting weights pattern duplication
 * @param scaling: Scaling number for reducing the weights pattern size
 * @param layer_index: Index of the layer
 * @return weights_mapping_Conv [scaling][IMCcol][IMCrow]
 */

void weights_mapping_Conv(CovLayer *cc, uint8_t ***VMM_weights_map, int *weights_number,
                          int scaling, int layer_index)
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
    {
        for (int i = 0; i < cc->input_channels; i++)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    if (layer_index == 1)
                        map_array[j][i][k2] = weights_map_1[j][i][x][y];
                    else
                        {
                            map_array[j][i][k2] = weights_map_2[j][i][x][y];
                        }
                    k2++;
                }
            }

            for (int y = 0; y < 9/2; y++)
            {
                int temp = map_array[j][i][y];
                map_array[j][i][y] = map_array[j][i][9-1-y];
                map_array[j][i][9-1-y] = temp;
            }

            k2 = 0;
        }
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

                for (int _y = 0; _y < drift_y; _y++)
                {
                    for (int sch = 0; sch < scaling; sch++)
                    {
                        VMM_weights_map[sch][h + drift_x][_y] = 0;
                    }
                }
            }

            if (layer_index > 1)
            {

                for (int i = 0; i < 9 * (input_channels); i++)
                {
                    // for (int sch = 0; sch < scaling; sch++)
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
                                // printf("@%d,%d->%d ", h + drift_x, _index_x_test_0 - _row_index_test / 2-1, mid_map_array[h][_index_x_test_0]);
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
                // printf("\n");
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

            for (int i = 9 * (input_channels / scaling) + drift_y; i < IMCrow; i++)
            {
                for (int sch = 0; sch < scaling; sch++)
                {
                    VMM_weights_map[sch][h + drift_x][i] = 0;
                }
            }
            // printf("\n");
        }

        drift_x += 1 * output_channels;
        drift_y += 3 * input_channels / scaling;
    }

    // for (int i = 0; i < scaling; i++)
    // {
    //     for (int j = 0; j < IMCcol; j++)
    //     {
    //         for (int z = 0; z < IMCrow; z++)
    //             printf("%d ", VMM_weights_map[i][j][z]);
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    /*counting patten duplication times*/
    for (int i = 0; i < IMCcol; i++)
    {
        if (((i + 1) % output_channels == 0) && (i > 1))
            (*weights_number)++;
    }
}

/**
 * @brief Mapping weights into 32x36 matrix for fully connected layer.
 * It updates in each scaling turn
 *
 * @param fc Pointer to the fully connected layer.
 * @param VMM_weights_map Pointer to the 3D array that stores the weight map.
 * @param layer_index The index of the layer.
 * @param scaling The scaling factor.
 */
void weights_mapping_FC(OutputLayer *fc, uint8_t ***VMM_weights_map,
                        int layer_index, int scaling)

{

    int input_channels = fc->input_num;
    int output_channels = fc->output_num;

    for (int i = 0; i < IMCcol; i++)
    {
        for (int j = 0; j < IMCrow; j++)
        {
            if (layer_index == 3)
            {
                VMM_weights_map[0][i][j] = weights_map_3[j + scaling * IMCrow][i];
            }
            else
            {
                if (j < fc->input_num && i < fc->output_num)
                    VMM_weights_map[0][i][j] = weights_map_4[j + scaling * IMCrow][i];
                else
                    VMM_weights_map[0][i][j] = 0;
            }
        }

    }

    /*debug printing*/
    // if(layer_index==3)
    // {
    //     for(int i=0;i<IMCcol; i++)
    //     {
    //         for(int j=0; j<IMCrow; j++)
    //         {
    //             printf("%d ", VMM_weights_map[0][i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    
}

/**
 *@brief Creates 9x1 lines of image data and concatenates lines into a 2D array.
 *
 *@param cc ConvLayer struct representing the convolutional layer.
 *@param images List of input images.
 *@param maplist List of mapped inputs, represented as a scal x VMM_turns array.
 *@param VMM_turns Number of VMM pages.
 *@param scaling Scaling factor for the input, dividing input mapping into "scaled" pieces
 *@param layer_index Index of the layer, for changing the strategy of mapping
 *@note For the Conv1 layer, the input mapping will have half space empty at each 4th ([3]) page.
 */
void inputs_mapping_Conv(CovLayer *cc, uint8_t ***images, uint8_t ***maplist, int *VMM_turns,
                         int scaling, int layer_index)

{
    MatSize temp_input_size;
    temp_input_size.columns = cc->input_width;
    temp_input_size.rows = cc->input_height;
    printf("temp_input_size.column:%d\n", temp_input_size.columns);
    printf("temp_input_size.rows:%d\n", temp_input_size.rows);

    /*convert 2D array to 1D*/
    uint8_t VMM_input[IMCrow] = {0};
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
    int back_step = 0; // distance between the top of VMM to the highest kernel
    if (layer_index == 1)
        back_step = 6;

    printf("scaling:%d\n", scaling);
    printf("base_index_x:%d\n", base_index_x);

    for (int scal = 0; scal < scaling; scal++)
    {
        count_y = 0;
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
                            VMM_input[count_x] = images[ch + scaling * scal][r][c]; // go through the image by channles and scaling
                            // printf("%d->%d,%d  ", count_x, r, c);
                            // printf("->%d  ", VMM_input[count_x]);
                            count_x++; // increment index in one column (36)
                        }
                        index_VMM_input += cc->input_channels / scaling;
                        if (r >= columns_number - 1 && index_VMM_input < IMCrow - back_step && layer_index == 1)
                        {
                            index_VMM_input = IMCrow - back_step - 1;
                        }
                        if (index_VMM_input > IMCrow - back_step) // when input has been enough for one page
                        {
                            // printf("index_VMM_input: %d, r: %d ", index_VMM_input,r );
                            index_VMM_input = 1;
                            if (r < columns_number - 1) // if filter moves on the way
                            {
                                if (layer_index == 1)
                                    r -= 2 / (cc->input_channels / scaling); // duplicate four rows of the previous
                                else
                                    r -= 4 / (cc->input_channels / scaling); // duplicate four rows of the previous
                                printf("@@\n");

                                for (int h = 0; h < size_xx; h++)
                                {
                                    _local_VMM_input_lists[scal][count_y][h] = VMM_input[h];
                                }

                                for (int _input = 0; _input < IMCrow; _input++)
                                    VMM_input[_input] = 0;

                                count_y++;
                            }

                            else // if filter touches the end of one column
                            {
                                /*once it moved to the end(new column coming next page), new VMM created with only 4 rows mapped*/
                                printf("!!!\n");

                                /*store the current input array into list*/
                                for (int h = 0; h < size_xx; h++)
                                {
                                    _local_VMM_input_lists[scal][count_y][h] = VMM_input[h];
                                }

                                for (int _input = 0; _input < IMCrow; _input++)
                                    VMM_input[_input] = 0;

                                count_y++;
                            }
                            // printf("@@!!@@");
                            // for (int h = 0; h < size_xx; h++)
                            // {
                            //     printf("%d ", _local_VMM_input_lists[scal][count_y][h]);
                            // }
                            // printf("@@!!@@\n");
                            count_x = 0; // move count_x = 0; out of if condition logic
                        }
                    }
                }
            }
            if ((r >= temp_input_size.columns) && (c >= temp_input_size.columns))
                break;
        }
        printf("scal: %d\n", scal);
        printf("index_VMM_input: %d\n", index_VMM_input);
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
                // printf("%d ", _local_VMM_input_lists[d][i][h]);
            }
            // printf("\n");
        }
        // printf("\n");
    }

    *VMM_turns = count_y;
}

/**

*@brief Map input images to input array for fully connected layer
*
*@param fc Pointer to the fully connected layer
*@param images Input images to be mapped
*@param maplist 1x1x(input_num) array that will store the input array for VMM
*@param VMM_turns Number of VMM pages
*@param scaling Scaling number
*@param layer_index Index of the current layer
*/
void inputs_mapping_FC(OutputLayer *fc, uint8_t ***images, uint8_t ***maplist, int *VMM_turns,
                       int scaling, int layer_index)
{
    if (layer_index == 3)
    {
        for (int i = 0; i < scaling; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                for (int z = 0; z < 6; z++)
                {
                    maplist[0][i][z + 6 * j] = images[i][j][z];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < fc->input_num; i++)
        {
            maplist[0][0][i] = images[0][0][i];
        }
    }
}

/**
*@brief This function imports the Convolutional Neural Network (CNN) from the header file.
*
*@param cnn A pointer to the CNN object to be imported.
*@param i The index of the layer.
*/
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

/**
*@brief Initializes the Vector Mateix Multiplication (VMM) for the given CNN.
*This function allocates memory for the VMM and sets its properties based on the input CNN. It also assigns the
*MACoperation function pointer to the VMM's function pointer. This function returns the initialized VMM.
*
*@param cnn A pointer to the CNN for which the VMM is being initialized.
*@return A pointer to the initialized VMM.
*/

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

/**
*@brief Performs the Multiply and Accumulate operation (MAC) for a convolutional layer.
*@param conv_layer Pointer to the convolutional layer.
*@param input_array Pointer to the input array.
*@param output_array Pointer to the output array.
*@param weight_array Pointer to the weight array.
*@param page_image The index of the current page.
*@param sc The current scaling factor.
*@param layer_index The index of the current layer.
*/
void MACoperation(CovLayer *conv_layer, uint8_t ***input_array, uint8_t ***output_array, uint8_t ***weight_array,
                  int page_image, int sc, int layer_index)
/*why the convolution layer output zero padding has only 4 rows???*/
{
    float fweight = 0;
    float fimage = 0;
    uint8_t uweight = 0;
    uint8_t uimage = 0;
    float dotproduct = 0;
    /*input array [78][36] weight_array[32][36]*/
    for (int h = 0; h < IMCcol; h++)
    /*loop for 32 times in each column*/
    {
        /*parallel MAC will end here (I hope so!)*/
        for (int d = 0; d < IMCrow; d++)
        /*loop for 36 times in each row*/
        { // shape mismatch might be caused by precision issue
            fweight = bin_float_for_image_weights(weight_array[sc][h][d], 1);
            // printf("%d->", weight_array[sc][h][d]);
            fimage = bin_float_for_image_weights(input_array[sc][page_image][d], 0);
            // printf("%d ", input_array[sc][page_image][d]); // for fully connected layer
            dotproduct += fweight * fimage;
        }
        // printf("\n");
        // if (h % 4 == 0)
        output_array[sc][page_image][h] = float_bin_for_result(dotproduct);
        // printf("%.4f  ", dotproduct);
        dotproduct = 0;
    }
    // printf("\n");

    // for (int sc = 0; sc < scaling; sc++)
    // {
    //     // for (int i = 0; 0< i < VMM_turns; i++)
    //     {
    //         // printf("calculate %d \n", i);
    //         // for (int h = 0; h < IMCcol; h++)
    //         /*loop for 32 times in each column*/
    //         {
    //             /*parallel MAC will end here (I hope so!)*/
    //             for (int d = 0; d < IMCrow; d++)
    //             /*loop for 36 times in each row*/
    //             {
    //                 // fweight = bin_float_for_image_weights(weight_array[0][h][d], 1);
    //                 fimage = bin_float_for_image_weights(input_array[sc][page_image][d], 0);
    //                 // if (fweight != 0)
    //                 // printf("%.2f ", fimage);
    //                 printf("%d ", input_array[sc][page_image][d]);
    //             }
    //             printf("\n");
    //         }
    //     }
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

    // {
    //     printf("sc: %d, page: %d \n", sc, page_image);
    //     for (int r = 0; r < IMCcol; r++)
    //         printf("%d ", output_array[sc][page_image][r]);
    //     printf("\n");
    // }
}

/**

*@brief Performs occumulation operation on the results array and activate the result in the convolution layer.
*
*@param conv_layer Pointer to the convolution layer.
*@param pool_layer Pointer to the pooling layer.
*@param input_array Three dimensional array that contains the input data.
*@param VMM_turns Number of times the VMM is applied on the input data.
*@param weights_number Number of weights used in the convolution operation.
*@param scaling Number of scaling factors used in the convolution operation.
*@param layer_index Index of the current layer.
*/
void Conv_image(CovLayer *conv_layer, PoolingLayer *pool_layer, uint8_t ***input_array,
                int VMM_turns, int weights_number, int scaling, int layer_index)
{
    int in_channel_number = conv_layer->input_channels;
    int out_channel_number = conv_layer->output_channels;
    /*in each VMM turns*/
    int row_index = 0;
    int column_index = 0;
    int leftover_number = pool_layer->input_width % weights_number;
    float mac_in_process = 0;
    float mac_in_end = 0;
    int d = 0;
    int count = 0;
    int zero_limit1 = 16;       // for ReLu to exclude all low values magic number
    int zero_limit2 = 16;
    int page_at_columnend = 4; // replace formular out_channel_number / scaling - (scaling - 1)
    if (layer_index == 2)
        page_at_columnend = 3;
    printf("pool_layer->input_width: %d\n", pool_layer->input_width);
    printf("leftover_number: %d\n", leftover_number);
    printf("weights_number: %d\n", weights_number);
    int index = 0;
    int bias_1_layer[4] = {0};
    int bias_2_layer[8] = {0};

    if (layer_index == 1)
    {
        for (int i = 0; i < VMM_turns; i++)
        {
            if (((i + 1) % page_at_columnend == 0) && (i > 1)) // I really don't know what the fuck it is
            /* when it comes to end of coulmns*/
            {
                for (int h = 0; h < leftover_number * out_channel_number; h++) // here it defines that only half of resultlist will be taken
                {
                    /*for each scanning x 4*/
                    if (((h + 1) % out_channel_number == 0) && (h > 1))
                    {
                        for (d = 0; d < out_channel_number; d++)
                        {
                            /*assign value from i VMM turn for dth channel, ith column, h element*/
                            for (int scl = 0; scl < scaling; scl++)
                            {
                                mac_in_end = bin_float_for_result(input_array[scl][i][h + d - conv_layer->output_channels + 1]);
                            }
                            // printf("leftover_number:%d, h:%d,  d:%d", leftover_number, h, d);
                            // printf("%d  ", h + d - conv_layer->output_channels + 1);
                            // printf("@@row_: %d, column_: %d ", row_index, column_index);

                            mac_in_end += bin_float_for_bias(bias_1[d]);
                            // printf("bias:  %f\n", bin_float_for_bias(bias_1[d]));
                            if (mac_in_end < 0)
                                mac_in_end = 0;

                            conv_layer->v[d][row_index][column_index]  = float_bin_for_result(mac_in_end);
                            // if (conv_layer->v[d][row_index][column_index] <= zero_limit1)
                            //     conv_layer->v[d][row_index][column_index] = zero_limit1;

                            conv_layer->v[d][row_index][column_index] -= zero_limit1;
                            int temp = conv_layer->v[d][row_index][column_index] * 16;
                            if (temp > 255)
                                temp = 255;
                            conv_layer->v[d][row_index][column_index] = temp;

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
                // printf("\n");
                // printf("\n");

                row_index = 0;
                column_index++;
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
                            { // why here is -1 it is wrong
                                mac_in_process = bin_float_for_result(input_array[scl][i][h + d - conv_layer->output_channels + 1]);
                            }
                            // printf("row_: %d,column_: %d ", row_index, column_index);
                            mac_in_process += bin_float_for_bias(bias_1[d]);
                            // printf("bias:  %f\n", bin_float_for_bias(bias_1[d]));
                            if (mac_in_process < 0)
                                mac_in_process = 0;

                            conv_layer->v[d][row_index][column_index] = float_bin_for_result(mac_in_process);
                            // if (conv_layer->v[d][row_index][column_index] <= zero_limit1)
                                // conv_layer->v[d][row_index][column_index] = zero_limit1;

                            conv_layer->v[d][row_index][column_index] -= zero_limit1;
                            int temp = conv_layer->v[d][row_index][column_index] * 16;
                            // if (mac_in_process > 4)
                                // printf("%f-> %d  ", mac_in_process, temp);
                            if (temp > 255)
                                temp = 255;
                            conv_layer->v[d][row_index][column_index] = temp;

                            // convolution might have some issues here
                            // printf("row_:%d, column_:%d  ", row_index, column_index);
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
                // printf("\n");
            }
        }
    }
    else
    {
        for (int i = 0; i < VMM_turns; i++)
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
                            mac_in_end += bin_float_for_result(input_array[scl][i][h + d - conv_layer->output_channels + 1]);
                        }
                        // printf(" @@@row_: %d,column_: %d ", row_index ,column_index);

                        mac_in_end += bin_float_for_bias(bias_2[d]);
                        if(mac_in_end < 0)
                            mac_in_end = 0;
                        conv_layer->v[d][row_index][column_index] = float_bin_for_result(mac_in_end);
                        // if (conv_layer->v[d][row_index][column_index]!=32)
                        // if (conv_layer->v[d][row_index][column_index] <= zero_limit2)
                            // conv_layer->v[d][row_index][column_index] = zero_limit2;

                        conv_layer->v[d][row_index][column_index] -= zero_limit2;
                        int temp = conv_layer->v[d][row_index][column_index] * 16;
                        if (temp > 255)
                            temp = 255;
                        conv_layer->v[d][row_index][column_index] = temp;
                        // printf("row_:%d, column_:%d  ", row_index, column_index);
                        mac_in_end = 0;
                    }
                    if (row_index == conv_layer->input_width - conv_layer->map_size)
                    {
                        row_index = 0;
                    }
                    else
                        row_index++; // printf("finished\n");
                }
            }
            if (((i + 1) % page_at_columnend == 0) && (i > 1)) // I really don't know what the fuck it is
            /* when it comes to end of coulmns*/
            {
                row_index = 0;
                column_index++;
            }

        // printf("\n");
        // printf("\n");
        }

        // MatSize ySize = {12, 12};
        // conv_layer->v[0]=  MatRotate180(conv_layer->v[0], ySize);
        // conv_layer->v[1] = MatRotate180(conv_layer->v[1], ySize);

        // conv_layer->v[4] = MatRotate180(conv_layer->v[4], ySize);
        // conv_layer->v[5] = MatRotate180(conv_layer->v[5], ySize);

        // uint8_t inter_array[12][12] = {0};

        // for(int d=0; d< 12; d++)
        // {
        //     for(int i=0; i<12; i++)
        //         inter_array[d][i] = conv_layer->v[1][12 - d - 1][i];
        // }

        // for (int d = 0; d < 12; d++)
        // {
        //     for (int i = 0; i < 12; i++)
        //         conv_layer->v[1][d][i] = inter_array[d][i];
        // }


        // for(int d=0; d< 12; d++)
        // {
        //     for(int i=0; i<12; i++)
        //         inter_array[d][i] = conv_layer->v[4][12 - d - 1][i];
        // }

    }
}

/**
 * @brief Accumulate the results lists for fully-connected layer and stores the result as the output 
 * 
 * @param fc_layer Pointer to the fc layer
 * @param input_array Three-dimensional array holding the input data, 1x1x32 in O5, and 1X1X10 in O6
 * @param scaling The scaling factor used for quantization
 * @param layer_index Index of the layer
 */
void FC_image(OutputLayer *fc_layer, uint8_t ***input_array,
              int scaling, int layer_index)
{
    uint8_t zero_limit = 16;
    for (int i = 0; i < fc_layer->output_num; i++)
    {
        float mac_result = 0;
        float bias_ = 0;
        for (int j = 0; j < scaling; j++)
        {
            mac_result += bin_float_for_result(input_array[0][j][i]);
        }
        if (layer_index == 3)
        {
            bias_ = bin_float_for_bias(bias_3[i]);
        }
        else
        {
            bias_ = bin_float_for_bias(bias_4[i]);
        }
        float activated_result = mac_result * 0.5 + bias_;
        // printf("activated_result: %d   ", (int)(activated_result * 100));
        if (activated_result < 0)
            activated_result = 0;
        fc_layer->v[i] = float_bin_for_result(activated_result); // 0.5 is the propotion of output from python model
        if (fc_layer->v[i] <= zero_limit)
        {
            fc_layer->v[i] = zero_limit;
        }
        fc_layer->v[i] -= zero_limit;
        int temp = fc_layer->v[i]*16;
        if(temp>255)
            temp = 255;
        fc_layer->v[i] = temp;
        // printf("v[i]: %d  ", fc_layer->v[i]);
    }
    // printf("\n");
}

/**
*@brief print the 2D array of image data
*
*@param scale The scale of the image
*@param image_data The 2D array of image data
*/
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

/**
*@brief Frees memory space of a Convolutional Layer
*This function frees the memory space of a Convolutional Layer by freeing
*the memory of each element of the 3D matrix v[j][r], and then freeing
*the memory of the array of pointers v[j] pointing to the 2D matrices,
*and finally freeing the memory of the array of pointers v pointing to the
*array of pointers v[j].
*
*@param covL The Convolutional Layer to free its memory space
*/
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
        }
        free(covL->v[j]);
    }
    free(covL->v);
    free(covL);
}

/**
*@brief Frees the memory of a given pooling layer.
*
*@param pol Pointer to the pooling layer to be freed.
*/
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

/**
*@brief Frees the memory space of a fully connected output layer.
*
*@param FC Pointer to the fully connected output layer to free the memory space of.
*/
void freeFClayer(OutputLayer *FC)
/*free space of Convolutional layer*/
{
    printf("freeFCLayer!\n");
    int i, j, c, r;
    int outH = FC->output_num;
    free(FC->v);
    free(FC);
}

/**
 * @brief Generates a 3D array for input data
 *
 * @param scal The number of input images
 * @param size The width and height of each image
 * @return uint8_t*** The generated 3D array
 */
uint8_t ***generate_input_array(int scal, int size)
{
    uint8_t ***VMM_input_array = alloc_3darray(scal, size, IMCrow);

    return VMM_input_array;
}

/**
 * @brief Generates a 3D array to hold the results
 *
 * @param scal The number of input images
 * @param VMM_turns The number of times VMM mac operation needs to be applied
 * @return uint8_t*** The generated 3D array
 */
uint8_t ***generate_result_array(int scal, int VMM_turns)
{
    uint8_t ***result_list = alloc_3darray(scal, VMM_turns, IMCcol);

    return result_list;
}

/**
 * @brief Frees the memory allocated for a 3D array
 *
 * @param data The 3D array to free
 * @param xlen The length of the x dimension
 * @param ylen The length of the y dimension
 */
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

/**
 * @brief Frees the memory allocated for a 2D array
 *
 * @param data The 2D array to free
 * @param xlen The length of the x dimension
 */
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

/**
 * @brief Frees the memory allocated for an array of images
 *
 * @param image_array The array of images to free
 * @param rows The number of rows in each image
 * @param number_of_images The number of images to free
 */
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

/**
*@brief Allocates a 3D array with dimensions xlen, ylen, and zlen.
*
*@param xlen The length of the first dimension.
*@param ylen The length of the second dimension.
*@param zlen The length of the third dimension.
*@return A pointer to the allocated 3D array.
*/
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

/**
*@brief Allocates a 2D array with dimensions xlen and ylen.
*
*@param xlen The length of the first dimension.
*@param ylen The length of the second dimension.
*@return A pointer to the allocated 2D array.
*/
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

/**
*@brief Allocates a 1D array with length xlen.
*
*@param xlen The length of the array.
*@return A pointer to the allocated 1D array.
*/
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
