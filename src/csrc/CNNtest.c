// // Main function of CNN Train and Test.
// //

// #include <stdlib.h> //random
// #include <stdio.h>
// #include <time.h> //random seed
// #include <assert.h>
// #include "cnn.h"

// #define AVG_POOLING 0 // Pooling with average
// #define MAX_POOLING 1 // Pooling with Maximum
// #define MIN_POOLING 2 // Pooling with Minimum

// float** weigths_mapping(Cnn *cnn);
// float** inputs_mapping(float **inputdata);
// void _CnnFF(Cnn *cnn, float **input_data);
// void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size);
// ImageArray _ReadImages(const char *filename);


// int main()
// {
//     // Read train and test data.
//     // LabelArray test_labels = ReadLabels("mnist/t10k-labels-idx1-ubyte");//???
//     ImageArray test_images = _ReadImages("mnist/t10k-images-idx3-ubyte");//???
//     printf("[+] Read data finished!\n");

//     // Input image mat size {columns,rows}{28,28}
//     MatSize input_size;//
//     input_size.columns = test_images->image_point[0].number_of_columns;
//     input_size.rows = test_images->image_point[0].number_of_rows;
//     printf("[+] Input size: {%d,%d}\n", input_size.columns, input_size.rows);

//     // Output Label array size {label_length} {10}
//     int output_size = 10;
//     // int output_size = test_labels->label_point[0].label_length;
//     printf("[+] Output size: %d\n", output_size);

//     // Setup CNN
//     Cnn *cnn = (Cnn *)malloc(sizeof(Cnn));//???
//     _CnnSetup(cnn, input_size, output_size);
//     printf("[+] CNN setup finished!\n");

//     printf("[+] Import CNN finished!\n");
//     int test_images_num = 100;
//     float incorrect_ratio = 1.0;

//     float **weigthts =  weigths_mapping(cnn);
//     float **imagedata = inputs_mapping(test_images->image_point[0].image_data);

//     _CnnFF(cnn, test_images->image_point[0].image_data);


//     return 0;
// }

// void _CnnSetup(Cnn *cnn, MatSize input_size, int output_size)
// {
//     int map_size = 5;
//     cnn->layer_num = 5; // layers = 5
//     int pool_scale = 2;

//     MatSize temp_input_size;

//     // Layer1 Cov input size: {28,28}
//     temp_input_size.columns = input_size.columns;
//     temp_input_size.rows = input_size.rows;
//     cnn->C1 = InitialCovLayer(temp_input_size.columns,
//                               temp_input_size.rows, map_size, 1, 4, SAME);

//     // Layer2 Pooling input size: {28,28}
//     temp_input_size.columns = temp_input_size.columns - map_size + 1;
//     temp_input_size.rows = temp_input_size.rows - map_size + 1;
//     cnn->S2 = InitialPoolingLayer(temp_input_size.columns,
//                                   temp_input_size.rows, pool_scale, 4, 4, MAX_POOLING);

// }

// void _CnnFF(Cnn *cnn, float **input_data)
// /*
//     1st Convolution + Pooling
// */
// {
//     int output_sizeW = cnn->S2->input_width;
//     int output_sizeH = cnn->S2->input_height;
//     MatSize map_size = {cnn->C1->map_size, cnn->C1->map_size};
//     MatSize input_size = {cnn->C1->input_width, cnn->C1->input_height};
//     MatSize output_size = {cnn->S2->input_width, cnn->S2->input_height};



//     /*convolution result is cnn->C1->v*/
//     for (int i = 0; i < (cnn->C1->output_channels); i++)
//     {
//         /*Activation function with params of weighted input and bias*/
//         for (int row = 0; row < output_size.rows; row++)
//             for (int col = 0; col < output_size.columns; col++)
//                 cnn->C1->y[i][row][col] = ActivationReLu(cnn->C1->v[i][row][col],
//                                                          cnn->C1->basic_data[i]);
//     }
//     output_size.columns = cnn->S2->input_width / 2;
//     output_size.rows = cnn->S2->input_height / 2;
//     input_size.columns = cnn->S2->input_width;
//     input_size.rows = cnn->S2->input_height;
//     for (int i = 0; i < (cnn->S2->output_channels); i++)
//     {
//         if (cnn->S2->pooling_type == AVG_POOLING)
//             AvgPooling(cnn->S2->y[i], output_size, cnn->C1->y[i],
//                        input_size, cnn->S2->map_size);
//         else if (cnn->S2->pooling_type == MAX_POOLING)
//             MaxPooling(cnn->S2->y[i], output_size, cnn->C1->y[i],
//                        input_size, cnn->S2->map_size);
//     }
// }

// // Read one image from data <filename>
// ImageArray _ReadImages(const char *filename)
// {
//     // Open images file
//     FILE *file_point = NULL;
//     file_point = fopen(filename, "rb");
//     if (file_point == NULL) // Failed
//     {
//         printf("[-] ReadImages() Open file [%s] failed!\n", filename);
//         assert(file_point);
//     }

//     // Read images from file with file_point
//     int magic_number = 0;     // magic number
//     int number_of_images = 0; // Images' number
//     int n_rows = 0;           // number of rows of an image<image hight>
//     int n_columns = 0;        // number of cols of an image<image width>

//     // >Big-End Style, So Reverse the Integer. Read magic number
//     fread((char *)&magic_number, sizeof(magic_number), 1, file_point);
//     magic_number = ReverseInt(magic_number);

//     // >Big-End. Read the number of images.
//     fread((char *)&number_of_images, sizeof(number_of_images), 1, file_point);
//     number_of_images = ReverseInt(number_of_images);

//     // Read the rows and cols of an image
//     fread((char *)&n_rows, sizeof(n_rows), 1, file_point);
//     fread((char *)&n_columns, sizeof(n_columns), 1, file_point);
//     n_rows = ReverseInt(n_rows);
//     n_columns = ReverseInt(n_columns);

//     // define strutrue of image array
//     ImageArray image_array = (ImageArray)malloc(sizeof(ImageArray));
//     image_array->number_of_images = number_of_images; // number of images
//     // array of all images.
//     image_array->image_point = (MnistImage *)malloc(number_of_images * sizeof(MnistImage));

//     int row, column;                           // Temp for row and column

//     image_array->image_point[0].number_of_rows = n_rows;       //
//     image_array->image_point[0].number_of_columns = n_columns; // set
//     image_array->image_point[0].image_data = (float **)malloc(n_rows * sizeof(float *));
    
//     for (row = 0; row < n_rows; ++row) // from 0 -> n_rows-1
//     {
//         image_array->image_point[0].image_data[row] = (float *)malloc(n_columns * sizeof(float));
//         for (column = 0; column < n_columns; ++column) // from 0 -> n_columns-1
//         {
//             unsigned char temp_pixel = 0;
//             // read a pixel 0-255 with 8-bit
//             fread((char *)&temp_pixel, sizeof(temp_pixel), 1, file_point);
//             // Change 8-bit pixel to float.
//             image_array->image_point[0].image_data[row][column] = (float)temp_pixel / 255;
//         }
//     }

//     fclose(file_point);
//     return image_array;
// }

// float **weigths_mapping(Cnn *cnn)
// {

//     printf("below is weighs map\n");
//     /*convert 2D map to 1D*/
//     float map_array[1][4][9];
//     int k2 = 0;
//     for (int j = 0; j < (cnn->C1->input_channels); j++)
//         for (int i = 0; i < (cnn->C1->output_channels); i++)
//         {

//             for (int x = 0; x < 3; x++)
//             {
//                 for (int y = 0; y < 3; y++)
//                 {
//                     map_array[j][i][k2] = cnn->C1->map_data[j][i][x][y];
//                     printf("%f ", map_array[j][i][k2]);
//                     k2++;
//                 }
//             }
//             k2 = 0;
//             printf("\n");
//         }

//     /* in the future to map the weights like this*/
//     /*convert 1D array to 2D*/
//     printf("weights rotate 90 degree\n");
//     static float weights_map[4][9];
//     for (int j = 0; j < (cnn->C1->input_channels); j++)
//     {
//         for (int d = 0; d < 9; d++)
//         {
//             for (int i = 0; i < (cnn->C1->output_channels); i++)
//             {
//                 weights_map[i][d] = map_array[j][i][d];
//                 printf("%f ", weights_map[i][d]);
//             }
//             printf("\n");
//         }
//         printf(";\n");
//     }
//     return weights_map;
// }

// float **inputs_mapping(float **input_data)
// {
//         printf("below is image map\n");

//         /*convert 2D array to 1D*/
//         float image_array[784];
//         static float image_map[18][32];
//         int k = 0;
//         for (int x = 0; x < 28; x++)
//             for (int y = 0; y < 28; y++)
//             {
//                 image_array[k] = input_data[x][y];
//                 k++;
//             }
//         /*take 18 pixels and print by 32 columns*/
//         /*convert 1D array to 2D*/
//         int drift_factor = 400;
//         for (int i = 0; i < 18; i++)
//         {
//             for (int h = 0; h < 32; h++)
//             {
//                 image_map[i][h] = image_array[i + drift_factor];
//                 printf("%.2f ,", image_map[i][h]);
//             }
//             printf(";\n");
//         }

//         return image_map;
// }
