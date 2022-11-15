// Main function of CNN Train and Test.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> //random seed
#include <assert.h>
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
/*map weigths and iamge as in IMC shape*/
    int VMM_turns = 0;
    float **input_data_array = inputs_mapping(&(test_images->image_point[0]), input_size, &VMM_turns);
    float** weight_array = weigths_mapping(cnn);
/*initalize VMM and use MAC operation*/
    VMM* vmm = initializeVMM(cnn);
    float **output_array = vmm->MACoperation(input_data_array, weight_array, VMM_turns);

    /*after convolution result from ADC*/
    // _CnnFF(cnn, test_images->image_point[0].image_data);


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
                              temp_input_size.rows, map_size, 1, 4, SAME);

    // Layer2 Pooling input size: {28,28}
    temp_input_size.columns = temp_input_size.columns - map_size + 1;
    temp_input_size.rows = temp_input_size.rows - map_size + 1;
    cnn->S2 = InitialPoolingLayer(temp_input_size.columns,
                                  temp_input_size.rows, pool_scale, 4, 4, MAX_POOLING);
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
    if (file_point == NULL) // Failed
    {
        printf("[-] ReadImages() Open file [%s] failed!\n", filename);
        assert(file_point);
    }

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
    image_array->number_of_images = 1; // number of images
    // array of all images.
    image_array->image_point = (MnistImage *)malloc(1 * sizeof(MnistImage));

    int row, column; // Temp for row and column

    image_array->image_point[0].number_of_rows = n_rows;       //
    image_array->image_point[0].number_of_columns = n_columns; // set
    image_array->image_point[0].image_data = (float **)malloc(n_rows * sizeof(float *));

    for (row = 0; row < n_rows; ++row) // from 0 -> n_rows-1
    {
        image_array->image_point[0].image_data[row] = (float *)malloc(n_columns * sizeof(float));
        for (column = 0; column < n_columns; ++column) // from 0 -> n_columns-1
        {
            unsigned char temp_pixel = 0;
            // read a pixel 0-255 with 8-bit
            fread((char *)&temp_pixel, sizeof(temp_pixel), 1, file_point);
            // Change 8-bit pixel to float.
            image_array->image_point[0].image_data[row][column] = (float)temp_pixel / 255;
        }
    }

    fclose(file_point);
    return image_array;
}

float **weigths_mapping(Cnn *cnn)
{

    printf("below is weighs map\n");
    /*convert 2D map to 1D*/
    float **VMM_weights_map;
    VMM_weights_map = malloc(sizeof(float*)*IMCcol);
    for(int i = 0; i<IMCcol; i++)
        VMM_weights_map[i] = malloc(sizeof(float*)*IMCrow);

    float map_array[1][4][9];
    int k2 = 0;
    int drift_x = 0;
    int drift_y = 0;
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

    // /*debug*/
    // for (int i = 0; i < sizeof(VMM_weights_map) / sizeof(VMM_weights_map[0]);i++)
    // {
    //     for (int h = 0; h < sizeof(VMM_weights_map[0]) / sizeof(VMM_weights_map[0][0]); h++)
    //     printf("%f ", VMM_weights_map[i][h]);

    //     printf("\n");

    // }

    return VMM_weights_map;
}

float **inputs_mapping(MnistImage *image, MatSize input_size, int* VMM_turns)
/*Create 9x1 lines of image data and concatenate lines into 2D array*/
{
    MatSize temp_input_size;
    temp_input_size.columns = input_size.columns;
    temp_input_size.rows = input_size.rows;

    printf("below is image map\n");

    /*convert 2D array to 1D*/
    int index_VMM_input_array = 0;
    float VMM_input[IMCrow];
    float _local_VMM_input_lists[1000][IMCrow];
    int count_x = 0;
    int count_y = 0;
    /*create image data metrix 3x3 for weights to associate*/
    int index_VMM_input = 1;
    int r = 0;
    int c = 0;
    for (int d = 0; d < 32; d++)
    {
        for (int i = 0; i < 30; i++)
        {
            for (r = 0 + i * 3; (r < 3 + i * 3) && (r < 28); r++)
            {
                for (c = 0 + d; (c < d + 3) && (c < 28); c++)
                {
                    // printf("%d,%d  ", r, c);
                    VMM_input[count_x] = image->image_data[r][c];
                    count_x++;
                    index_VMM_input++;
                    if(index_VMM_input >36)
                    {
                        index_VMM_input = 1;
                        if (r < image->number_of_columns - 1)
                            r -= 4;
                        // printf("!!\n\n");
                        count_x = 0;
                        for(int h=0; h<sizeof(VMM_input)/sizeof(VMM_input[0]); h++)
                            _local_VMM_input_lists[count_y][h] = VMM_input[h];
                        count_y++;
                    }
                }
            }
        }
            if((r>=27)&&(c>=28))
            break;
    }
    /*debug*/
    printf("\ncount_x: %d\n", count_x);
    printf("\ncount_y: %d\n", count_y);

    printf("length: %d\n", sizeof(VMM_input)/sizeof(VMM_input[0]));

    float **VMM_input_array;
    VMM_input_array = malloc(sizeof(float*)*count_y);
    for (int i = 0; i < count_y; i++)
    {
        VMM_input_array[i] = malloc(sizeof(float*)*IMCrow);
        for(int h=0; h < IMCrow; h++)
            VMM_input_array[i][h] = _local_VMM_input_lists[i][h];
    }
    *VMM_turns = count_y;
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

VMM* initializeVMM(Cnn *cnn)
{
    VMM *vmm = malloc(sizeof(VMM));
    vmm->Cnn = cnn;
    vmm->cols = IMCcol;
    vmm->rows = IMCrow;
    vmm->MACoperation = MACoperation;
    return vmm;
}

float** MACoperation(float** input_array, float** weight_array, int VMM_turns){
    /*input array [78][36] weight_array[32][36]*/
    float **output_array;

    output_array = malloc(sizeof(float *) * VMM_turns);
    for (int i = 0; i < VMM_turns; i++)
    {
        // printf("calculate %d \n", i);
        output_array[i] = malloc(sizeof(float*)*IMCcol);
        for (int h = 0; h < IMCcol; h++)
        /*loop for 32 times*/
        {
            for(int d=0; d< IMCrow; d++)
                /*loop for 36 times*/
                {
                    output_array[i][h] += input_array[i][d]*weight_array[h][d];
                }
        }
    }
    // /*debug*/
    // for (int d = 0; d < VMM_turns; d++)
    // {
    //     for (int r = 0; r < IMCcol; r++)
    //         printf("%f  ", output_array[d][r]);
    //     printf("\n");
    // }

    return output_array;
}
