// Main function of CNN Train and Test.
//

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include "basicTest.h"
#include "imagedata.h"
#include "weights_bias.h"

#define offset 0X3000100 //?
#define tx_status_adr 0x30001008
#define address_bus 0x30001004
#define info_bus 0x30001004
#define read

int main()
{
    printf("Welcome to A-Core for vmm test!\n");

    /*Prepare mapping*/
    // Read train and test data.
    // LabelArray test_labels = ReadLabels("mnist/t10k-labels-idx1-ubyte");
    ImageArray test_images = _ReadImages("mnist/t10k-images-idx3-ubyte");
    printf("[+] Read data finished!\n");

    // Input image mat size {columns,rows}
    MatSize input_size;
    input_size.columns = test_images->image_point[0].number_of_columns;
    input_size.rows = test_images->image_point[0].number_of_rows;

    printf("[+] Input size: {%d,%d}\n", input_size.columns, input_size.rows);

    // Output Label array size {label_length} {10}
    int output_size = 10;
    printf("[+] Output size: %d\n", output_size);
    // Setup CNN
    Cnn *cnn = (Cnn *)malloc(sizeof(Cnn));
    _CnnSetup(cnn, input_size, output_size);
    printf("[+] CNN setup finished!\n");

    // /*load weights from csv file*/
    _ImportCnn(cnn);
    printf("[+] Import CNN finished!\n");

    /*map weigths and iamge as in IMC shape*/
    int VMM_turns = 0;
    int weights_number = 0;
    int bias_number = 0;
    float ***weight_array = weights_mapping(cnn->C1, &weights_number, 1);
    // float *bias_array = bias_mapping(cnn->C1, &bias_number);

    printf("finish mapping of w, b\n");
    // freeConvLayer(cnn->C1);
    // free(cnn);

    printf("finish freeing space of w, b\n");

    /*remember to release the memory and produce cnn again after 1st MAC*/
    MnistImage *input_list[] = {&(test_images->image_point[0])};
    float **VMM_input_array;
    VMM_input_array = (float **)malloc(sizeof(float *) * 28);
    for (int i = 0; i < 28; i++)
    VMM_input_array[i] = (float *)malloc(sizeof(float) * IMCrow);

    printf("inputs_mapping!!!!");
    inputs_mapping(input_list, VMM_input_array, &VMM_turns, 1, 0);
    printf("@@@@finish mapping\n");


    for (int i = 0; i < 28; i++)
    {
        for (int h = 0; h < IMCrow; h++)
            printf("%f ", VMM_input_array[i][h]);
        printf("@@@@\n");
    }
    printf("@@@@scalling\n");


    return 0;
}

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

void _CnnFF(CovLayer *conv_layer, PoolingLayer *pool_layer, float **input_data)
/*
    1st Activation + Pooling
*/
{
    MatSize map_size = {conv_layer->map_size, conv_layer->map_size};
    MatSize input_size = {conv_layer->input_width, conv_layer->input_height};
    MatSize output_size = {pool_layer->input_width, pool_layer->input_height};
    int output_sizeW = pool_layer->input_width;
    int output_sizeH = pool_layer->input_height;

    /*convolution result is conv_layer->v*/
    for (int i = 0; i < (conv_layer->output_channels); i++)
    {
        /*Activation function with params of weighted input and bias*/
        for (int row = 0; row < output_size.rows; row++)
            for (int col = 0; col < output_size.columns; col++)
                conv_layer->y[i][row][col] = ActivationReLu(conv_layer->v[i][row][col],
                                                            conv_layer->basic_data[i]);
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
/*try to add padding from 28x28 to 30x30*/
{
    // Open images file
    FILE *file_point = NULL;
    file_point = fopen(filename, "rb");

    // Read images from file with file_point
    int magic_number = 0;     // magic number
    int number_of_images = 0; // Images' number
    int n_rows = 0;           // number of rows of an image<image hight>
    int n_columns = 0;        // number of cols of an image<image width>

    // >Big-End Style, So Reverse the Integer. Read magic number
    fread((char *)&magic_number, sizeof(magic_number), 1, file_point);
    magic_number = ReverseInt(magic_number);

    // >Big-End. Read the number of images.
    fread((char *)&number_of_images, sizeof(number_of_images), 1, file_point);
    number_of_images = ReverseInt(number_of_images);

    // Read the rows and cols of an image
    fread((char *)&n_rows, sizeof(n_rows), 1, file_point);
    fread((char *)&n_columns, sizeof(n_columns), 1, file_point);
    n_rows = ReverseInt(n_rows);
    n_columns = ReverseInt(n_columns);

    // define strutrue of image array
    ImageArray image_array = (ImageArray)malloc(sizeof(ImageArray));
    image_array->number_of_images = number_of_images; // number of images
    // array of all images.
    image_array->image_point = (MnistImage *)malloc(number_of_images * sizeof(MnistImage));

    int row, column;                           // Temp for row and column
    for (int i = 0; i < number_of_images; ++i) // Images from 0 -> number_of_images-1
    {
        image_array->image_point[i].number_of_rows = n_rows+2;       //
        image_array->image_point[i].number_of_columns = n_columns+2; // set
        image_array->image_point[i].image_data = (float **)malloc((n_rows+2) * sizeof(float *));


        /* adding 1x30 zero padding in the begining 0th row*/
        /*first row*/
        image_array->image_point[i].image_data[0] = (float *)malloc((n_columns + 2) * sizeof(float));

        for (column = 0; column < n_columns+2; ++column) // from 0 -> n_columns
        {
            unsigned char zero_pixel = 0;
            // Change 8-bit pixel to float.
            image_array->image_point[i].image_data[0][column] = (float)zero_pixel / 255;
        }

        /* adding 1x30 zero padding in the end row*/
        /*end row*/
        image_array->image_point[i].image_data[n_columns+1] = (float *)malloc((n_columns + 2) * sizeof(float));

        for (column = 0; column < n_columns + 2; ++column) // from 0 -> n_columns
        {
            unsigned char zero_pixel = 0;
            // Change 8-bit pixel to float.
            image_array->image_point[i].image_data[n_columns+1][column] = (float)zero_pixel / 255;
        }

        /*from 1st -> Nth rows*/
        for (row = 1; row < n_rows + 1; ++row) 
        {
            image_array->image_point[i].image_data[row] = (float *)malloc((n_columns + 2) * sizeof(float)); // expanding to 30
            
            /*adding zero padding in the begining col 0*/
            unsigned char zero_pixel = 0;
            image_array->image_point[i].image_data[row][0] = (float)zero_pixel / 255;

            for (column = 1; column < n_columns + 1; ++column) // from 0 -> n_columns
            {
                // unsigned char temp_pixel = 0;
                // // read a pixel 0-255 with 8-bit
                // fread((char *)&temp_pixel, sizeof(temp_pixel), 1, file_point);
                // // Change 8-bit pixel to float.
                // image_array->image_point[i].image_data[row][column] = (float)temp_pixel / 255;
                image_array->image_point[i].image_data[row][column] = (float)myarray[row][column];

            }
            
            /*adding zero padding in the end col+1*/
            image_array->image_point[i].image_data[row][n_columns + 1] = (float)zero_pixel / 255;
        }

    }

    fclose(file_point);
    return image_array;
}

float *bias_mapping(CovLayer *cc, int *bias_number)
{
    printf("below is bias map\n");
    float* VMM_bias_map;
    VMM_bias_map = malloc(sizeof(float) * IMCcol * cc->input_channels);
    for (int r = 0; r < IMCcol / cc->output_channels; r++)
        for (int i = 0; i < cc->output_channels; i++)
        {
            VMM_bias_map[r * cc->output_channels + i] = cc->basic_data[i];
        }

    /*debug*/
    // printf("bias@@@@\n");
    // for (int chan = 0; chan < cc->input_channels; chan++)
    // {
    //     for (int i = 0; i < IMCcol;i++)
    //     {
    //         printf("%.5f ", VMM_bias_map[i]);
    //     }
    //     printf("\n");
    // }

    printf("debug bias@@@@@\n");
    for(int chan=0; chan< cc->output_channels; chan++)
    {
        printf("%.5f ", cc->basic_data[chan]);
    }
    printf("\n");
    return VMM_bias_map;
}

float ***weights_mapping(CovLayer *cc, int *weights_number, int Scaling)
/*
    mapping weights into 32*36 matrix 
    param cc: convLayer for current layer
    param weights_number: pointer to numer counting weights pattern duplication
    param Scaling: scaling number for reduce the weights pattern size
*/
{

    int answer_in = 0;  
    int page_weight = 0;    
    int index_weight = 0;   
    float feed = 0; 
    int number = 0;

    printf("below is weights map\n");
    printf("inputchannel: %d\n", cc->input_channels);
    printf("outputchannel: %d\n", cc->output_channels);

    /*convert 2D map to 1D, memory acclocate space*/
    float ***VMM_weights_map;
    VMM_weights_map = malloc(sizeof(float **) * Scaling);

    for (int sch = 0; sch < Scaling; sch++)
    {
        VMM_weights_map[sch] = malloc(sizeof(float *) * IMCcol);
        for (int i = 0; i < IMCcol; i++)
        {
            VMM_weights_map[sch][i] = malloc(sizeof(float) * IMCrow);
        }
    }

    int input_channels = cc->input_channels;
    int output_channels = cc->output_channels;
    int map_num = cc->map_size * cc->map_size; 

    float map_array[output_channels][input_channels][map_num];
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
    float mid_map_array[output_channels][input_channels*map_num];

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

            for(int i=0; i< drift_y; i++){
                for (int sch = 0; sch < Scaling; sch++)
                {
                    VMM_weights_map[sch][h + drift_x][i] = 0;
                }
            }

            for (int i = 0; i < 9 * (input_channels / Scaling); i++)
            {
                for (int sch = 0; sch < Scaling; sch++)
                {
                    // for (int in_c = 0; in_c < cc->input_channels / Scaling; in_c++)
                    // {
                    //     i = i + in_c;
                        VMM_weights_map[sch][h + drift_x][i + drift_y] = mid_map_array[h][row_index];
                        row_index++;
                        // }
                }
            }

            for (int i = 9 * (input_channels / Scaling) + drift_y; i < IMCrow; i++)
            {
                for (int sch = 0; sch < Scaling; sch++)
                {
                        VMM_weights_map[sch][h + drift_x][i] = 0;
                }
            }
        }
        drift_x += 1 * output_channels;
        drift_y += 3 * input_channels / Scaling;
    }

    /*counting patten duplication times*/
    for (int i = 0; i < IMCcol; i++)
    {
        if (((i + 1) % output_channels == 0) && (i > 1))
            (*weights_number)++;
    }

    /*debug*/
    // printf("weights@@@ \n");
    // for(int sch=0; sch<Scaling; sch++)
    // {
    //     for (int h = 0; h < IMCcol;h++)
    //     {
    //         for (int i = 0; i < IMCrow; i++)
    //         {
    //             printf("%.2f ", VMM_weights_map[sch][h][i]);
    //         }
    //         if (((h + 1) % output_channels == 0) && (h > 1))
    //         {
    //             printf("!!\n");
    //         }
    //         else
    //             printf("\n");
    //     }
    //     printf("@@@@@\n");
    // }

    for (int i = 2 * 36; i < 34 * 36; i++)
    {
        page_weight = (int)(i / 36) - 2;
        index_weight = i % 36;
        feed = VMM_weights_map[0][page_weight][index_weight];
        number = converfloat_int8(feed, 1);
        printf("%d: %d\n", i, number);
    }

    return VMM_weights_map;
}

void inputs_mapping(MnistImage **images,float **maplist, int *VMM_turns, int scaling, int part_index)
/*Create 9x1 lines of image data and concatenate lines into 2D array*/
/*
    return: 3D array with image elements locations in 3x3 metrix
*/
{
    MatSize temp_input_size;
    temp_input_size.columns = images[0]->number_of_columns;
    temp_input_size.rows = images[0]->number_of_rows;

    // printf("below is image map\n");

    /*convert 2D array to 1D*/
    int index_VMM_input_array = 0;
    float VMM_input[IMCrow];
    int size_xx = sizeof(VMM_input) / sizeof(VMM_input[0]);
    float _local_VMM_input_lists[scaling][200][IMCrow];
    int count_x = 0;
    int count_y = 0;
    int index_VMM_input = 1;
    int columns_number = temp_input_size.columns;
    int _conved_size = temp_input_size.columns - 3 + 1;
    float temp_input[size_xx];
    int r = 0;
    int c = 0;

    /*could be for test only when there is no data feeding*/
    // cc->input_channels = 2;
    // cc->output_channels = 8;
    // temp_input_size.columns = 14;
    // temp_input_size.rows = 14;
    // columns_number = temp_input_size.rows;
    int base_index_x = IMCcol / 4 + 1;
    int base_index_y = IMCcol;

    printf("scaling:%d\n", scaling);

    for (int scal = 0; scal < scaling; scal++)
        for (int d = 0; d < _conved_size; d++) // base number for index in y direction(0:31)
        {
            for (int i = 0; i < base_index_x + 1; i++) // base number for index in x direction from 0 to 9 in one page (0:1:9)
            {
                for (r = 0 + i * 3 * 1 / scaling;                                                       // initial state
                     (r < 3 * 1 / scaling + i * 3 * 1 / scaling) && (r < temp_input_size.columns); r++) // index by x direction in one VMM page (0:1:11):(8,8,8):(16:1:27)
                {
                        for (c = 0 + d;                                      // initial state
                             (c < d + 3) && (c < temp_input_size.rows); c++) // index by y direction in one vmm page[(0, 1, 2):[3, 3, 3]:(25, 26, 27)]
                        {
                            /*collect image data into input array*/
                            for (int ch = 0; ch < 1 / scaling; ch++)
                            {
                                VMM_input[count_x] = images[ch]->image_data[r][c];
                                count_x++;
                            }

                            // printf("index_VMM_input: %d\n", index_VMM_input);
                            index_VMM_input = index_VMM_input + 1 / scaling;
                            // printf("index_VMM_input: %d\n", index_VMM_input);

                            if (index_VMM_input > IMCrow)
                            {
                                index_VMM_input = 1;
                                if (r < temp_input_size.columns - 1)
                                {
                                    r -= 4 / (1 / scaling);
                                    count_x = 0;

                                    // printf("normal count_y: %d\n", count_y);
                                    assign_to_sub_array(maplist, VMM_input, size_xx, count_y);
                                    count_y++;
                                }
                                else
                                {
                                    /*once it moved to the end, new VMM created with only 4 rows mapped*/
                                    // printf("!!!\n");
                                    count_x = 0;
                                    /*store the current input array into list*/
                                    // printf("store new count_y: %d\n", count_y);
                                    assign_to_sub_array(maplist, VMM_input, size_xx, count_y);

                                    for (int h = 0; h < size_xx; h++)
                                    {
                                        temp_input[h] = 0;
                                    }

                                    /*duplicate four lines from previous line*/
                                    for (int h = 0; h < size_xx; h++)
                                        if ((h < 4))
                                            temp_input[h] = VMM_input[size_xx + h - 4];

                                    count_y++;

                                    // printf("next count_y: %d\n", count_y);
                                    assign_to_sub_array(maplist, temp_input, size_xx, count_y);
                                    /*go to next line*/
                                    count_y++;

                                }
                                if ((count_y % ((part_index + 1) * 28) == 0) && (count_y > (part_index * 28)))
                                    /*use this  as a range swtich!*/
                                    return 0;
                            }
                        }
                }
            }
            if ((r >= temp_input_size.columns) && (c >= temp_input_size.columns))
                break;
        }

    *VMM_turns = count_y;
}

void quantize_data(float number, int bit){
    number = 2.4;
    printf("!!number: %f\n", number);
    float a;
    int b, k, l = 0, m[20], n, i, count = 0, p;
    b = (int)number;
    a = number - b;

    // while(b!=0)
    // {
        printf("!!b: %d\n", b);
        n=b%2;
        b/=2;
        m[i]=n;
        i++;
        count++;
    // }

    for(int k=0; k=count-1; k++)
    {
        printf("%d", m[k]);
    }

    for(int i=0; i<bit; i++)
    {
        a=a*2;
        l=a;
        if(l==1)
        a=a-1;
    }
    printf("\n");
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

// // import cnn from file
// void _ImportCnn(Cnn *cnn, const char *filename)
// {
//     FILE *file_point = NULL;
//     file_point = fopen(filename, "rb");

//     if (file_point == NULL)
//         printf("[-] <ImportCnn> Open file failed! <%s>\n", filename);

//     /*Loading C1 weights and bias, C1 has been initialized in _Convsetup*/
//     load_weights(file_point, cnn->C1);
//     load_bias(file_point, cnn->C1);

//     load_weights(file_point, cnn->C3);
//     load_bias(file_point, cnn->C3);

//     fclose(file_point);
// }

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

void load_weights(CovLayer *cc, float ****weights)
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

void load_bias(CovLayer *cc, float *bias)
{
    for (int ch = 0; ch < cc->output_channels; ch++)
    {
        cc->basic_data[ch] = bias[ch];
    }
}

// void load_weights(FILE *file_point, CovLayer *cc)
// {
//     printf("call@@@@\n");
//     char line[1024];
//     int i = 0; // input_channel index
//     for (int i = 0; i < cc->output_channels; i++)
//         for (int j = 0; j < cc->input_channels; j++)
//         {
//             for (int r = 0; r < cc->map_size; r++)
//             {
//                 do
//                 {
//                     fgets(line, 1024, file_point);
//                 } while (strcmp(line, "\n") == 0 || strcmp(line, "\n\n") == 0);
//                 for (int c = 0; c < cc->map_size; c++)
//                 {
//                     int h = c + 1;
//                     char *tmp = strdup(line);
//                     char *value = getfield(tmp, h);
//                     if (value != NULL)
//                     {
//                         float number = strtod(value, NULL);
//                         cc->map_data[i][j][r][c] = number;
//                     }
//                     free(tmp);
//                 }
//             }

//         }

// }
// void load_bias(FILE *file_point, CovLayer *cc)
// {
//     char line[1024];
//     int count_ = 0;
//     for (int i = 0; i < 1; i++)
//     {
//         do
//         {
//             fgets(line, 1024, file_point);
//             for (int ch = 0; ch < cc->output_channels; ch++)
//             {
//                 int h = ch + 1;
//                 char *tmp = strdup(line);
//                 char *value = getfield(tmp, h);
//                 if (value != NULL)
//                 {
//                     float number = strtod(value, NULL);
//                     cc->basic_data[count_] = number;
//                     count_++;
//                 }
//                 free(tmp);
//             }
//         } while (strcmp(line, "\n") == 0 || strcmp(line, "\n\n") == 0);
//         if (count_ < cc->output_channels)
//         {
//             i--;
//         }
//     }

// }

MnistImage *Output_image(int cols, int rows, float **imagedata)
/*convert output into MnistImage structure*/
{
    MnistImage *imagemodel = malloc(sizeof(MnistImage));
    imagemodel->image_data = imagedata;
    imagemodel->number_of_columns = cols;
    imagemodel->number_of_rows = rows;
    return imagemodel;
}


VMM *initializeVMM(Cnn *cnn)
/*initalize the VMM and return VMM*/
{
    VMM *vmm = malloc(sizeof(VMM));
    vmm->Cnn = cnn;
    vmm->cols = IMCcol;
    vmm->rows = IMCrow;
    vmm->MACoperation = MACoperation;
    return vmm;
}

float ***MACoperation(float ***input_array, float ***weight_array, int VMM_turns, int Scaling)
{
    /*input array [78][36] weight_array[32][36]*/
    float ***output_array;
    output_array = malloc(sizeof(float**)*Scaling);
    for(int sc=0; sc<Scaling;sc++)
    {
        output_array[sc] = malloc(sizeof(float *) * VMM_turns);
        for (int i = 0; i < VMM_turns; i++)
        {
            // printf("calculate %d \n", i);
            output_array[sc][i] = malloc(sizeof(float *) * IMCcol);
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

void Conv_image(CovLayer *conv_layer, PoolingLayer *pool_layer, float ***input_array, int VMM_turns, int weights_number, int scaling)
{
    int channels_number = conv_layer->output_channels;
    /*in each VMM turns*/
    int column_index = 0;
    int row_index = 0;
    int leftover_number = pool_layer->input_width % weights_number;

    for (int i = 0; i < VMM_turns; i++)
        if (((i + 1) % channels_number == 0) && (i > 1))
        /* when it comes to end of coulmns*/
        {
            for (int h = 0; h < leftover_number * channels_number; h++)
            {
                /*for each scanning x 4*/
                if (((h + 1) % channels_number == 0) && (h > 1))
                {
                    for (int d = 0; d < channels_number; d++)
                    {
                        /*assign value from i VMM turn for dth channel, ith column, h element*/
                        for(int scl=0;scl<scaling;scl++)
                        {
                            conv_layer->v[d][row_index][column_index] += input_array[scl][i][h + d];
                        }
                    }
                    row_index++;
                }
            }

            row_index = 0;
            column_index++;
        }
        else
        /*when it is on the way*/
        {
            for (int h = 0; h < IMCcol; h++)
            {
                /*for each scanning x 4*/
                if (((h + 1) % channels_number == 0) && (h > 1))
                {
                    for (int d = 0; d < channels_number; d++)
                    {
                        /*assign value from i VMM turn for dth channel, ith column, h element*/
                        for (int scl = 0; scl < scaling; scl++)
                        {
                            conv_layer->v[d][row_index][column_index] += input_array[scl][i][h + d];
                        }
                            // printf("d: %d, column_index: %d, ", )
                        }
                    row_index++;
                }
            }
        }
}

void save_image(int scale, float **image_data, const char *filename)
{
    FILE *filepoint = NULL;
    int temp = 0;
    filepoint = fopen(filename, "wb");

    // Writing Magic Number to the File
    fprintf(filepoint, "P2\n");

    // Writing Width and Height
    fprintf(filepoint, "%d %d\n", scale, scale);

    // Writing the maximum gray value
    fprintf(filepoint, "255\n");
    int count = 0;
    for (int i = 0; i < scale; i++)
    {
        for (int j = 0; j < scale; j++)
        {
            temp = (int)(image_data[i][j] * 255);

            // Writing the gray values in the 2D array to the file
            fprintf(filepoint, "%d ", temp);
        }
        fprintf(filepoint, "\n");
    }
    fclose(filepoint);
}

void freeConvLayer(CovLayer *col)
{
    col->is_full_connect = true; //
    int i, j, c, r;
    for (i = 0; i < col->output_channels; i++)
    {
        for (j = 0; j < col->input_channels; j++)
        {
            for (r = 0; r < col->map_size; r++)
            {
                free(col->map_data[i][j][r]);
            }
            free(col->map_data[i][j]);
        }
        free(col->map_data[i]);
    }
    free(col->map_data);

    free(col->basic_data);

    int outW = col->input_width - col->map_size + 1;
    int outH = col->input_height - col->map_size + 1;

    // covL->d = (float ***)malloc(output_channels * sizeof(float **));
    // covL->y = (float ***)malloc(output_channels * sizeof(float **));
    for (j = 0; j < col->output_channels; j++)
    {
        // covL->d[j] = (float **)malloc(outH * sizeof(float *));
        // covL->y[j] = (float **)malloc(outH * sizeof(float *));
        for (r = 0; r < outH; r++)
        {
            // covL->d[j][r] = (float *)calloc(outW, sizeof(float));
            free(col->v[j][r]);
            // covL->y[j][r] = (float *)calloc(outW, sizeof(float));
        }
        free(col->v[j]);
    }
    free(col->v);
    free(col);
}

void assign_to_sub_array(float **maplist, float *temp_input, int size_xx, int count_y)
{
    int sub_array_index = count_y / 28;
    for (int h = 0; h < size_xx; h++)
        maplist[count_y - sub_array_index * 28][h] = temp_input[h];
}

int converfloat_int8(float number, int isweight)
{
/*simplified steps to conver float into int8*/
    int answer_in = 0;
    float base = 0.015625;//pow(2,-6)
    answer_in = (int)((number)/base);
    if (answer_in < 0);
        answer_in *= -1;
    if(isweight){
        if (number>0)
            answer_in += 64;//pow(2,6)
        if(number<0)
            answer_in += 128;//pow(2,7)
    }
    return answer_in;
}