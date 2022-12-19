// Main function of CNN Train and Test.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "basicTest.h"

int main()
{
    const char *cnn_arch_filename = "output/param_decoded.csv";
    // Read train and test data.
    // LabelArray test_labels = ReadLabels("mnist/t10k-labels-idx1-ubyte");//???
    ImageArray test_images = _ReadImages("mnist/t10k-images-idx3-ubyte"); //???
    printf("[+] Read data finished!\n");

    // Input image mat size {columns,rows}{28,28}
    MatSize input_size; //
    input_size.columns = test_images->image_point[0].number_of_columns;
    input_size.rows = test_images->image_point[0].number_of_rows;
    printf("[+] Input size: {%d,%d}\n", input_size.columns, input_size.rows);

    // Output Label array size {label_length} {10}
    int output_size = 10;
    // int output_size = test_labels->label_point[0].label_length;
    printf("[+] Output size: %d\n", output_size);

    // Setup CNN
    Cnn *cnn = (Cnn *)malloc(sizeof(Cnn)); //???
    _CnnSetup(cnn, input_size, output_size);
    printf("[+] CNN setup finished!\n");

    printf("[+] Import CNN finished!\n");
    int test_images_num = 100;
    float incorrect_ratio = 1.0;

    /*load weights from csv file*/
    _ImportCnn(cnn, cnn_arch_filename);

    /*save data as image*/
    char *filename = (char *)malloc(sizeof(char) * 13);

    /*map weigths and iamge as in IMC shape*/
    int VMM_turns = 0;
    int weights_number = 0;
    int bias_number = 0;
    float **input_data_array = inputs_mapping(&(test_images->image_point[0]), input_size, &VMM_turns, cnn->C1->input_channels, cnn->C1->output_channels);
    float **weight_array = weights_mapping(cnn, &weights_number);
    float *bias_array = bias_mapping(cnn, &bias_number);

    // /*initalize VMM and use MAC operation*/
    // VMM *vmm = initializeVMM(cnn);
    // float **output_array = vmm->MACoperation(input_data_array, weight_array, VMM_turns);

    // Conv_image(cnn, output_array, VMM_turns, weights_number);

    // /*after convolution result from ADC*/
    // _CnnFF(cnn, test_images->image_point[0].image_data);

    // /*2nd convolution*/
    // int map_size = 3;
    // MatSize input_2nd_size;
    // input_2nd_size.columns = (input_size.columns-map_size+1)/2;
    // input_2nd_size.rows = (input_size.rows-map_size+1)/2;

    // input_data_array

    /*debug: create pgm files, later use convert in terminal to create png*/
    // sprintf(filename, "image_%d.pgm", 1);
    // save_image(cnn->C1->input_height,
    //         test_images->image_point[0].image_data,
    //         filename);
    // for(int i=0; i<cnn->C1->output_channels; i++)
    // {
    //     sprintf(filename, "conv_%d.pgm", i);
    //     save_image(cnn->S2->input_height, cnn->C1->v[i], filename);
    // }

    // for(int i=0;i<cnn->S2->output_channels; i++)
    // {
    //     sprintf(filename, "pool_%d.pgm", i);
    //     save_image(cnn->S2->input_height/2, cnn->S2->y[i], filename);
    // }

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

    cnn->C1 = InitialCovLayer(temp_input_size.columns,
                              temp_input_size.rows, map_size, 1, 4, VALID);

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

}

void _CnnFF(Cnn *cnn, float **input_data)
/*
    1st Activation + Pooling
*/
{
    int output_sizeW = cnn->S2->input_width;
    int output_sizeH = cnn->S2->input_height;
    MatSize map_size = {cnn->C1->map_size, cnn->C1->map_size};
    MatSize input_size = {cnn->C1->input_width, cnn->C1->input_height};
    MatSize output_size = {cnn->S2->input_width, cnn->S2->input_height};

    /*convolution result is cnn->C1->v*/
    for (int i = 0; i < (cnn->C1->output_channels); i++)
    {
        /*Activation function with params of weighted input and bias*/
        for (int row = 0; row < output_size.rows; row++)
            for (int col = 0; col < output_size.columns; col++)
                cnn->C1->y[i][row][col] = ActivationReLu(cnn->C1->v[i][row][col],
                                                         cnn->C1->basic_data[i]);
    }
    output_size.columns = cnn->S2->input_width / 2;
    output_size.rows = cnn->S2->input_height / 2;
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
}

// Read one image from data <filename>
ImageArray _ReadImages(const char *filename)
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
        image_array->image_point[i].number_of_rows = n_rows;       //
        image_array->image_point[i].number_of_columns = n_columns; // set
        image_array->image_point[i].image_data = (float **)malloc(n_rows * sizeof(float *));

        for (row = 0; row < n_rows; ++row) // from 0 -> n_rows-1
        {
            image_array->image_point[i].image_data[row] = (float *)malloc(n_columns * sizeof(float));
            for (column = 0; column < n_columns; ++column) // from 0 -> n_columns-1
            {
                unsigned char temp_pixel = 0;
                // read a pixel 0-255 with 8-bit
                fread((char *)&temp_pixel, sizeof(temp_pixel), 1, file_point);
                // Change 8-bit pixel to float.
                image_array->image_point[i].image_data[row][column] = (float)temp_pixel / 255;
            }
        }
    }

    fclose(file_point);
    return image_array;
}

float* bias_mapping(Cnn* cnn, int *bias_number)
{
    printf("below is bias map\n");
    float *VMM_bias_map;
    VMM_bias_map = malloc(sizeof(float)*IMCcol);
    for (int r = 0; r < IMCcol / cnn->C1->output_channels; r++)
        for (int i = 0; i < cnn->C1->output_channels; i++)
            VMM_bias_map[r * 4 + i] = cnn->C1->basic_data[i];

    // /*debug*/
    // for (int i = 0; i < IMCcol;i++)
    // {
    //     printf("%.2f ", VMM_bias_map[i]);
    //     printf("\n");
    // }

    return VMM_bias_map;
}

float **weights_mapping(Cnn *cnn, int *weights_number)
{

    printf("below is weights map\n");
    /*convert 2D map to 1D*/
    float **VMM_weights_map;
    VMM_weights_map = malloc(sizeof(float *) * IMCcol);
    for (int i = 0; i < IMCcol; i++)
        VMM_weights_map[i] = malloc(sizeof(float ) * IMCrow);

    float map_array[1][4][9];
    int k2 = 0;
    int drift_x = 0;
    int drift_y = 0;
    /*weights map for columns by output channels*/
    for (int j = 0; j < (cnn->C1->input_channels); j++)
        for (int i = 0; i < (cnn->C1->output_channels); i++)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    map_array[j][i][k2] = cnn->C1->map_data[j][i][x][y];
                    k2++;
                }
            }
            k2 = 0;
        }
    /*for mapping through the IMC matrix*/
    for (int r = 0; r < IMCcol / cnn->C1->output_channels; r++)
    {
        for (int h = 0; h < cnn->C1->output_channels; h++)
        {
            for (int i = 0; i < 9; i++)
            {
                VMM_weights_map[h + drift_x][i + drift_y] = map_array[0][h][i];
            }
        }
        drift_x += 4;
        drift_y += 3;
    }

    for (int i = 0; i < IMCcol; i++)
    {
        if (((i + 1) % 4 == 0) && (i > 1))
            (*weights_number)++;
    }

/*create file for test*/
    // FILE *fpt;
    // fpt = fopen("weight_array_0.csv", "w+");
    // for (int i = 0; i < IMCcol; i++)
    // {
    //     for (int r = 0; r < IMCrow; r++)
    //     {
    //         fprintf(fpt, "%f ", VMM_weights_map[i][r]);
    //     }
    //     fprintf(fpt, "\n");
    // }

    // fclose(fpt);

    /*debug*/
    // for (int i = 0; i < IMCcol;i++)
    // {
    //     for (int h = 0; h < IMCrow; h++)
    //     printf("%.2f ", VMM_weights_map[i][h]);
    //     if(((i+1)%4==0)&& (i > 1))
    //     {
    //         printf("!!\n");
    //     }
    //     printf("\n");

    // }

    return VMM_weights_map;
}

float **inputs_mapping(MnistImage *image, MatSize input_size, int *VMM_turns, int num_inchan, int num_outchan)
/*Create 9x1 lines of image data and concatenate lines into 2D array*/
/*
    return: 3D array with image elements locations in 3x3 metrix
*/
{
    MatSize temp_input_size;
    temp_input_size.columns = input_size.columns;
    temp_input_size.rows = input_size.rows;

    printf("below is image map\n");

    /*convert 2D array to 1D*/
    int index_VMM_input_array = 0;
    float VMM_input[IMCrow];
    int size_xx = sizeof(VMM_input) / sizeof(VMM_input[0]);
    float _local_VMM_input_lists[1000][IMCrow];
    int count_x = 0;
    int count_y = 0;
    int index_VMM_input = 1;
    int columns_number = image->number_of_columns;
    int r = 0;
    int c = 0;

    num_inchan = 1;
    int base_index_x = IMCcol / (num_inchan * num_outchan) + 1;
    int base_index_y = IMCcol;

    for (int d = 0; d < 32; d++) // base number for index in y direction(0:31)
    {
        for (int i = 0; i < 10; i++) // base number for index in x direction from 0 to 9 in one page (0:1:9)
        {
            for (r = 0 + i * 3;                    // initial state
                 (r < 3 + i * 3) && (r < 28); r++) // index by x direction in one VMM page (0:1:11):(8,8,8):(16:1:27)
            {
                printf("r: %d\n", r);
                for (c = 0 + d;                    // initial state
                     (c < d + 3) && (c < 28); c++) // index by y direction in one vmm page[(0, 1, 2):[3, 3, 3]:(25, 26, 27)]
                {
                    /*collect image data into input array*/
                    // printf("%d,%d ", r, c);
                    VMM_input[count_x] = image->image_data[r][c];
                    count_x++;
                    index_VMM_input++;
                    if (index_VMM_input > 36)
                    {
                        index_VMM_input = 1;
                        if (r < columns_number - 1)
                        {
                            r -= 4;
                            // printf("\n");
                            count_x = 0;
                            for (int h = 0; h < size_xx; h++)
                                _local_VMM_input_lists[count_y][h] = VMM_input[h];
                            count_y++;
                        }
                        else
                        {
                            /*once it moved to the end, new VMM created with only 4 rows mapped*/
                            // printf("!!!\n");
                            count_x = 0;
                            /*store the current input array into list*/
                            for (int h = 0; h < size_xx; h++)
                                _local_VMM_input_lists[count_y][h] = VMM_input[h];

                            float temp_input[size_xx];
                            /*duplicate four lines from previous line*/
                            for (int h = 0; h < size_xx; h++)
                                if ((h < 4))
                                    temp_input[h] = VMM_input[size_xx + h - 4];
                                else
                                    temp_input[h] = 0;

                            /*store the new input array into list*/
                            count_y++;
                            for (int h = 0; h < size_xx; h++)
                                _local_VMM_input_lists[count_y][h] = temp_input[h];
                            /*go to next line*/
                            count_y++;
                        }
                    }
                }
            }
        }
        if ((r >= 27) && (c >= 28))
            break;
    }
    /*debug*/
    printf("\ncount_x: %d\n", count_x);
    printf("\ncount_y: %d\n", count_y);
    printf("input size: %d\n", temp_input_size.columns);
    printf("base_index_x: %d\n", base_index_x);

    float **VMM_input_array;
    VMM_input_array = malloc(sizeof(float *) * count_y);
    for (int i = 0; i < count_y; i++)
    {
        VMM_input_array[i] = malloc(sizeof(float *) * IMCrow);
        for (int h = 0; h < IMCrow; h++)
            VMM_input_array[i][h] = _local_VMM_input_lists[i][h];
    }

    // /*create file for test*/
    // FILE *fpt;
    // fpt = fopen("image_array_0.csv", "w+");
    // for (int i = 0; i < count_y; i++)
    // {
    //     for (int r = 0; r < IMCrow; r++)
    //     {
    //         fprintf(fpt, "%f ", VMM_input_array[i][r]);
    //     }
    //     fprintf(fpt, "\n");
    // }

    // fclose(fpt);

    *VMM_turns = count_y;

    // for(int i=0; i<count_y; i++)
    // {
    // for(int h=0; h<size_xx; h++)
    //     printf("%f ",  _local_VMM_input_lists[i][h]);
    // printf("\n");
    // }

    return VMM_input_array;
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

// import cnn from file
void _ImportCnn(Cnn *cnn, const char *filename)
{
    FILE *file_point = NULL;
    file_point = fopen(filename, "rb");

    if (file_point == NULL)
        printf("[-] <ImportCnn> Open file failed! <%s>\n", filename);

    /*Loading C1 weights and bias, C1 has been initialized in _Convsetup*/
    load_weights(file_point, cnn->C1);
    load_bias(file_point, cnn->C1);

    fclose(file_point);
}

void load_weights(FILE *file_point, CovLayer *cc)
{
    char line[1024];
    for (int i = 0; i < cc->input_channels; i++)
        for (int j = 0; j < cc->output_channels; j++)
        {
            for (int r = 0; r < cc->map_size; r++)
            {
                do
                {
                    fgets(line, 1024, file_point);
                } while (strcmp(line, "\n") == 0 || strcmp(line, "\n\n") == 0);
                for (int c = 0; c < cc->map_size; c++)
                {
                    int h = c + 1;
                    char *tmp = strdup(line);
                    char *value = getfield(tmp, h);
                    if (value != NULL)
                    {
                        float number = strtod(value, NULL);
                        cc->map_data[i][j][r][c] = number;
                    }
                    free(tmp);
                }
            }
        }
}

void load_bias(FILE *file_point, CovLayer *cc)
{
    char line[1024];
    for (int i = 0; i < cc->input_channels; i++)
    {
        do
        {
            fgets(line, 1024, file_point);
        } while (strcmp(line, "\n") == 0 || strcmp(line, "\n\n") == 0);
        for (int i = 0; i < cc->output_channels; i++)
        {
            int h = i + 1;
            char *tmp = strdup(line);
            char *value = getfield(tmp, h);
            if (value != NULL)
            {
                float number = strtod(value, NULL);
                cc->basic_data[i] = number;
            }
            free(tmp);
        }
    }
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

float **MACoperation(float **input_array, float **weight_array, int VMM_turns)
{
    /*input array [78][36] weight_array[32][36]*/
    float **output_array;
    output_array = malloc(sizeof(float *) * VMM_turns);
    for (int i = 0; i < VMM_turns; i++)
    {
        // printf("calculate %d \n", i);
        output_array[i] = malloc(sizeof(float *) * IMCcol);
        for (int h = 0; h < IMCcol; h++)
        /*loop for 32 times in each column*/
        {
            for (int d = 0; d < IMCrow; d++)
            /*loop for 36 times in each row*/
            {
                output_array[i][h] += input_array[i][d] * weight_array[h][d];
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

void Conv_image(Cnn *cnn, float **input_array, int VMM_turns, int weights_number)
{
    int channels_number = cnn->C1->output_channels;
    /*in each VMM turns*/
    int column_index = 0;
    int row_index = 0;
    int leftover_number = cnn->S2->input_width % weights_number;

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
                        cnn->C1->v[d][row_index][column_index] = input_array[i][h + d];
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
                        cnn->C1->v[d][row_index][column_index] = input_array[i][h + d];
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