#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include "basicTest.h"

#ifdef ACORE
#include "a-core-utils.h"
#include "a-core.h"
#endif
/*please check input data form in python entity side*/
#define LAYER1_SPLIT 28

#ifdef ACORE
#define SRAM_DIN_CONTROL_ADDR A_CORE_AXI4LVMM + 0x04
#define SRAM_din A_CORE_AXI4LVMM + 0x05
#define SRAM_wr_rd_addr A_CORE_AXI4LVMM + 0x06
#define OUTPUT_ADC_READY A_CORE_AXI4LVMM + 0x08
#define OUTPUT_ADC A_CORE_AXI4LVMM + 0x0C
#endif

uint8_t C1_y[28][28] = {
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 32, 31, 34, 31, 33, 31, 32, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 33, 31, 42, 31, 43, 31, 38, 31, 35, 31, 35, 31, 35, 31, 35, 31, 32, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 33, 33, 51, 34, 52, 33, 49, 31, 45, 31, 45, 31, 45, 31, 45, 31, 38, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 34, 36, 48, 44, 49, 41, 54, 36, 54, 35, 55, 35, 54, 35, 56, 34, 47, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 32, 41, 36, 54, 37, 51, 43, 46, 47, 45, 48, 45, 48, 45, 54, 43, 49, 33, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 40, 31, 49, 31, 51, 32, 54, 34, 54, 35, 55, 36, 55, 50, 55, 45, 36, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 34, 31, 37, 31, 40, 31, 46, 31, 46, 31, 48, 35, 49, 54, 58, 39, 37, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 33, 31, 33, 32, 35, 41, 41, 56, 55, 34, 34, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 47, 45, 51, 51, 32, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 34, 34, 52, 51, 43, 46, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 36, 37, 54, 56, 37, 39, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 31, 40, 41, 53, 54, 32, 33, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 33, 32, 48, 45, 49, 49, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 36, 34, 54, 50, 42, 43, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 42, 38, 56, 54, 36, 37, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 33, 33, 48, 45, 50, 53, 32, 32, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 35, 34, 54, 52, 42, 47, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 31, 41, 38, 57, 57, 35, 39, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 48, 45, 57, 53, 34, 33, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 33, 34, 52, 53, 55, 48, 33, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 37, 31, 58, 31, 47, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 41, 31, 59, 31, 44, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 39, 31, 55, 31, 38, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
    {31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 36, 31, 42, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31}};

// #define CONVERT_INT8_FLOAT(X) (float)_I2F_LUT[X]
// #define CONVERT_FLOAT_INT8(x, isweight) (isweight ? (x > 0 ? ((int)(abs(x / 0.015625)) + 64) : ((int)(abs(x / 0.015625)) + 128)) : ((int)(abs(x / 0.015625))))

// // statically initialize some data in .data section
// const float _I2F_LUT[64] = {
//     -3.9375, -3.8125, -3.6875, -3.5625,
//     -3.4375, -3.3125, -3.1875, -3.0625,
//     -2.9375, -2.8125, -2.6875, -2.5625,
//     -2.4375, -2.3125, -2.1875, -2.0625,
//     -1.9375, -1.8125, -1.6875, -1.5625,
//     -1.4375, -1.3125, -1.1875, -1.0625,
//     -0.9375, -0.8125, -0.6875, -0.5625,
//     -0.4375, -0.3125, -0.1875, -0.0625,
//     0.0625, 0.1875, 0.3125, 0.4375,
//     0.5625, 0.6875, 0.8125, 0.9375,
//     1.0625, 1.1875, 1.3125, 1.4375,
//     1.5625, 1.6875, 1.8125, 1.9375,
//     2.0625, 2.1875, 2.3125, 2.4375,
//     2.5625, 2.6875, 2.8125, 2.9375,
//     3.0625, 3.1875, 3.3125, 3.4375,
//     3.5625, 3.6875, 3.8125, 3.9375};

// int convert_float_int8(float number, int isweight)
// {
//     /*simplified steps to conver uint8_t into int8*/
//     int answer_in = 0;
//     float base = 0.015625; // pow(2,-6)
//     answer_in = (int)((number) / base);
//     if (answer_in < 0)
//     {
//         answer_in *= -1;
//     }
//     if (isweight)
//     {
//         if (number > 0)
//             answer_in += 64; // pow(2,6)
//         if (number < 0)
//             answer_in += 128; // pow(2,7)
//     }
//     return answer_in;
// }

// int convert_int8_float(int feed)
// {
//     /*decode the numbers from int8 to uint8_t, skipping binary string part*/
//     float number = 0;
//     int answer_in = 0;
//     number = feed * 0.125 - 4 + 0.0625; // *pow(2, -3) - 4 + pow(2, -3);
//     answer_in = number * 10000;
//     return answer_in;
// }

#ifdef ACORE
void write(int address, uint8_t data)
{
    int high_bits = (int)(address / 256);
    int low_bits = address - high_bits * 256;

    volatile uint32_t info = 0;
    /*set write mode*/
    info |= ((volatile uint8_t)data << 16);
    /*avoid any moment to make VMM reset without order*/
    info |= (1UL << 14);  // wr
    info &= ~(1UL << 13); // rd
    info |= (volatile char)high_bits << 8;
    info |= (volatile uint8_t)low_bits << 0;

    gpo_write((volatile uint32_t *)(SRAM_DIN_CONTROL_ADDR), info);
}

#endif

#ifdef ACORE
void set_readmode()
{
    volatile uint32_t info = 0;
    info |= (1UL << 13);  // rd
    info &= ~(1UL << 14); // wr
    /*reset read and write mode*/
    gpo_write((volatile uint32_t *)(SRAM_DIN_CONTROL_ADDR), info); /*set read mode*/
}
#endif

#ifdef ACORE
volatile uint8_t read(int index)
{
    set_readmode();
    delay(1);
    // volatile uint32_t *gpi_addr = (volatile uint32_t *)(OUTPUT_ADC+index*4);
    volatile uint8_t info = (volatile uint8_t)gpi_read((volatile uint32_t *)(OUTPUT_ADC + index * 4));
    return info;
}
#endif

#ifdef ACORE
void reset_VMM()
{
    volatile uint32_t info = 0;
    // info |= (1UL << 12); // rd
    info &= ~(1UL << 14); // wr
    info &= ~(1UL << 13); // rd

    gpo_write((volatile uint32_t *)(A_CORE_AXI4LVMM + 0x4), info);
}
#endif

void FeedVMM_weights(uint8_t ***weight_array, int VMM_turns, int Scaling)
{
    int page_weight = 0;
    int index_weight = 0;
    int number = 0;
    uint8_t feed = 0; // do not change this please
    for (int i = 2 * IMCrow; i < (IMCrow - 2) * IMCrow; i++)
    {
        page_weight = (int)(i / IMCrow) - 2;
        index_weight = i % IMCrow;
        feed = weight_array[0][page_weight][index_weight];
        number = feed;
#ifdef ACORE
        write(i, number);
#endif
    }
}

void FeedVMM_image(uint8_t ***VMMarray, int pagenumber, int Scaling)
{
    int number = 0;
    uint8_t feed = 0;
    for (int sc = 0; sc < Scaling; sc++)
    {
        for (int i = 0; i < IMCrow; i++)
        {
            feed = VMMarray[0][pagenumber][i];
            number = feed;
#ifdef ACORE
            write(i, number);
#endif
        }
    }
}
void VMMMACoperation(uint8_t ***result_list, int pagenumber, int Scaling)
{
    int result = 0;
    for (int h = 0; h < IMCcol; h++)
    {
#ifdef ACORE
        result = (int)read(h);
        result_list[0][pagenumber][h] = (uint8_t)result;
#endif
        // printf("%d ", result);
    }
    // printf("\n");
}

void main()
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
    Cnn *cnn = malloc(sizeof(*cnn));
    _CnnSetup(cnn, input_size, output_size);
    printf("[+] CNN setup finished!\n");

    // // // // // /*load weights from csv file, not needed anymore*/
    // // // // _ImportCnn(cnn);
    // // // // printf("[+] Import CNN finished!\n");

    /*map weigths and iamge as in IMC shape*/

    int VMM_turns = 0;
    int weights_number = 0;
    int bias_number = 0;
    int page_image = 0;
    int h = 0;
    int temp = 0;
#ifdef ACORE
    uint8_t ***result_list = generate_result_array();
// uint8_t ***result_list;
// result_list = malloc(sizeof(*result_list) * 1);
// result_list[0] = malloc(sizeof(*result_list[0]) * 28);
// for (int i = 0; i < 28; i++)
// result_list[0][i] = malloc(sizeof(*result_list[0][i]) * IMCcol);
#endif

    uint8_t ***weight_array = weights_mapping(cnn->C1, &weights_number, 1);
    uint16_t *bias_array = bias_mapping(cnn->C1, &bias_number);
    int turn_number = 0;
    int column_dex = 0;

    MnistImage *input_list[] = {&(test_images->image_point[0])};
    // uint8_t ***VMM_input_array = generate_input_array();
    uint8_t ***VMM_input_array;
    VMM_input_array = malloc(sizeof(*VMM_input_array) * 1);
    VMM_input_array[0] = malloc(sizeof(*VMM_input_array[0]) * 28);
    for (int i = 0; i < 28; i++)
        VMM_input_array[0][i] = malloc(sizeof(*VMM_input_array[0][i]) * IMCrow);

/*VMM R/W*/
#ifdef ACORE
    reset_VMM(); // printf and byte transmit
#else
    VMM *vmm = initializeVMM(cnn);
#endif

#ifndef DEBUG
    printf("writing weights!\n");
    FeedVMM_weights(weight_array, VMM_turns, 1);
    printf("\n");
#endif
    /*foor loop by 4 here*/
    for (turn_number = 0; turn_number < 4; turn_number++)
    {
        /*remember to release the memory and produce cnn again after 1st MAC*/
        printf("inputs_mapping!!!!\n");
        printf("turn %d\n", turn_number);
        inputs_mapping(cnn->C1, input_list, VMM_input_array,
                       &VMM_turns, 1, turn_number);
        printf("@@@@finish mapping\n");
        /*image making values*/
        /*weights*/
        #ifdef ACORE
        for (page_image = 0; page_image < 28; page_image++)
        {

            printf("writing image\n");
            FeedVMM_image(VMM_input_array, page_image, 1);
            printf("reading!\n");
            VMMMACoperation(result_list, page_image, 1);
        }
        #else
            uint8_t ***result_list = vmm->MACoperation(VMM_input_array, weight_array, 28, 1);
        #endif
        Conv_image(cnn->C1, 28, result_list, VMM_turns, weights_number, 1, &column_dex);
    }

    // free_input_array(VMM_input_array);
    // free_result_array(result_list);

    save_image(28, cnn->C1->v[0]);

    // freeConvLayer(cnn->C1);
    // freePoolLayer(cnn->S2);
    // printf("free cnn!\n");
    // free(cnn);

#ifdef ACORE
    test_pass();
    test_end();
#endif
}

/*we could compare the Cov result with python model when the pool and RELU are implemented */
