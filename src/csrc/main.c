#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <time.h>
#include <stdint.h>
#include "basicTest.h"
#include "imagedata.h"
#include "out_s2_debug.h"

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

/**
*@brief Writes data to VMM at the specified address in write mode.
*
*@param address Address of the VMM to write to.
*@param data Data to write to VMM.
*/
void write_VMM(int address, uint8_t data)
{
#ifdef ACORE
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
#endif
}

/**
*@brief Sets the VMM to read mode.
*/
void set_readmode()
{
#ifdef ACORE
    volatile uint32_t info = 0;
    info |= (1UL << 13);  // rd
    info &= ~(1UL << 14); // wr
    /*reset read and write mode*/
    gpo_write((volatile uint32_t *)(SRAM_DIN_CONTROL_ADDR), info); /*set read mode*/
#endif
}

/**
*@brief Reads data from VMM at the specified index.
*
*@param index Index of the VMM to read from.
*@return volatile uint8_t Returns the value read from VMM.
*/
volatile uint8_t read_VMM(int index)
{
#ifdef ACORE
    set_readmode();
    delay(1);
    // volatile uint32_t *gpi_addr = (volatile uint32_t *)(OUTPUT_ADC+index*4);
    volatile uint8_t info = (volatile uint8_t)gpi_read((volatile uint32_t *)(OUTPUT_ADC + index * 4));
    return info;
#endif
}

/**
*@brief Resets the VMM to initial state.
*/
void reset_VMM()
{
#ifdef ACORE
    volatile uint32_t info = 0;
    // info |= (1UL << 12); // rd
    info &= ~(1UL << 14); // wr
    info &= ~(1UL << 13); // rd

    gpo_write((volatile uint32_t *)(A_CORE_AXI4LVMM + 0x4), info);
#endif
}

/**
*@brief Feeds the weights to VMM for a given scale.
*
*@param weight_array Pointer to the 3D weight array.
*@param scal Index of the scale.
*/
void FeedVMM_weights(uint8_t ***weight_array, uint8_t scal)
{
    // printf("\n");
    int column_weights = 0;
    int index_weight = 0;
    uint8_t feed = 0; // do not change this please
    for (int i = 2 * IMCrow; i < (IMCrow - 2) * IMCrow; i++)
    {
        column_weights = (int)(i / IMCrow) - 2;
        index_weight = i % IMCrow;
        feed = weight_array[scal][column_weights][index_weight];
        if (feed != 0)
        {
#ifdef ACORE
            write_VMM(i, feed);
#endif
        }
        // printf("%d ", feed);
    }
    // printf("\n");
}

/**
*@brief Feeds an input image to the VMM
*This function takes a 3D array representing an input image for the VMM and
*feeds it to the VMM one row at a time. It iterates over the rows of the
*image, and for each row, it writes the pixel data to the corresponding
*address in the VMM.
*
*@param VMMarray The 3D array representing the input image for the VMM
*@param pagenumber The index of the current image being fed to the VMM
*@param scal The current scale of the input image
*/
void FeedVMM_image(uint8_t ***VMMarray, uint8_t pagenumber, uint8_t scal)
{
    // printf("\n");

    uint8_t feed = 0;
    for (uint8_t i = 0; i < IMCrow; i++)
    {
        feed = VMMarray[scal][pagenumber][i];
#ifdef ACORE
        write_VMM(i, feed);
#endif
        // printf("%d  ", feed);
    }
    // printf("\n");
}

/**

*@brief Read the results at the end of each column in VMM for a layer
*This function performs the reading peration to compute the output of a
*layer. It reads the pixel data from the VMM one pixel at a time, performs
*the multiplication and accumulation operation, and stores the result in a
*3D array representing the output of the layer.

*@param result_list The 3D array representing the output of the layer
*@param pagenumber The index of the current image being processed
*@param scal The current scale of the image being processed
*/
void VMMMACoperation(uint8_t ***result_list, int pagenumber, int scal)
{
    uint8_t result = 0;
    for (int h = 0; h < IMCcol; h++)
    {
#ifdef ACORE
        result = (uint8_t)read_VMM(h);
#endif
        result_list[scal][pagenumber][h] = result;
        // printf("%d ", (int)(result_list[scal][pagenumber][h]));
    }
    // printf("\n");
}

/**
*@brief Converts a binary fixed-point number to a float
*This function takes a binary fixed-point number represented as a uint8_t and
*converts it to a float. The binary fixed-point number is assumed to have a
*signed 2-bit integer part and a signed 6-bit fractional part. The function
*returns a float that represents the same number.
*
*@param number The binary fixed-point number to convert
*@return float The equivalent floating-point representation of the number
*/

float bin_float_for_activation(uint8_t number)
/*this function return float number which can not be shown in ACORE!!!*/
{
    int sign = 1;
    float base = 0.0625; // 2**(-4))
    float answer_in = 0;
    int8_t _number = number - 63;

    // base = 0.03125; // 2**(-5)

    answer_in = (float)(_number * base * sign);
    if (answer_in > base * 8)
        answer_in -= base * 8;
    return answer_in;
}

void main()
{
/**
 * @brief Prints a welcome message for the A-Core vmm test
 */
    printf("Welcome to A-Core for vmm test!\n");

    /*Setup CNN*/
    Cnn *cnn = calloc(1, sizeof(*cnn));

    /*first convolution*/
    /*set up VMM*/
#ifdef ACORE
    reset_VMM();
#else
    VMM *vmm = initializeVMM(cnn);
#endif

    int VMM_turns = 0;
    int weights_number = 0;
    int turn_number = 0;
    int column_dex = 0;
    int scal = 1; // scal for the 1st conv, as 2 in the 2nd convolution
    int output_size = 10;

    printf("[+] Read data finished!\n");

    // Input image mat size {columns,rows}
    MatSize input_size;
    input_size.columns = 30;
    input_size.rows = 30;

    printf("[+] Input size: {%d,%d}\n", input_size.columns, input_size.rows);

    // Output Label array size {label_length} {10}
    printf("[+] Output size: %d\n", output_size);

    _CnnSetup(cnn, input_size, output_size, 1);
    printf("[+] CNN setup finished!\n");

    /*below are for the 1st convolution*/
    /*weights*/
    uint8_t ***weight_array = alloc_3darray(scal, IMCcol, IMCrow);
    weights_mapping_Conv(cnn->C1, weight_array, &weights_number, scal, 1);
    /*remember to release the memory and produce cnn again after 1st MAC*/
    uint8_t ***VMM_input_array = generate_input_array(scal, 112);
    uint8_t ***result_list = generate_result_array(scal, 112);
    uint8_t ***image_input = alloc_3darray(1, 30, 30);
    for (int i = 0; i < 30; i++)
        for (int j = 0; j < 30; j++)
            image_input[0][i][j] = myimagearray[0][i][j];

    /*image making values*/
    printf("inputs_mapping_Conv!!!!\n");
    inputs_mapping_Conv(cnn->C1, image_input, VMM_input_array, &VMM_turns, scal, 1);
    free_3darray(image_input, 1, 30);

    printf("@@@@finish mapping\n");
    for (uint8_t i = 0; i < scal; i++)
    /*foor loop by 4 here*/
    {
        reset_VMM(); // reset to clean VMM, so only non-zero weights are needed
        printf("writing weights!\n");
        FeedVMM_weights(weight_array, i);
        printf("\n");
        for (int page_image = 0; page_image < VMM_turns; page_image++)
        {
// printf("reading\n");
#ifdef ACORE
            FeedVMM_image(VMM_input_array, page_image, i);
            VMMMACoperation(result_list, page_image, i); // feedVMM and write VMMmac should be looped by scal right?
#else
            vmm->MACoperation(cnn->C1, VMM_input_array, result_list, weight_array, page_image, i, 1);
#endif
        }
        printf("\n");
    }
    printf("@@@@@@@@@@@@@Convimage: %d\n", VMM_turns);

    Conv_image(cnn->C1, cnn->S2, result_list, VMM_turns, weights_number, scal, 1);
    free_3darray(VMM_input_array, scal, VMM_turns);
    free_3darray(result_list, scal, VMM_turns);
    free_3darray(weight_array, scal, IMCcol);

    printf("save image!!\n");
    _CnnFF(cnn->C1, cnn->S2);

    #ifndef ACORE

        for (int ch_i = 0; ch_i < 4; ch_i++)
        {
            for (int i = 0; i < 14; i++)
            {
                for (int h = 0; h < 14; h++)
                    printf("%d ", *(cnn->S2->y[ch_i][i][h]));
                printf("\n");
            }
            printf("\n");
        }
    #else

        for (int ch_i = 0; ch_i < 4; ch_i++)
        {
            save_image(14, cnn->S2->y[ch_i]);
            printf("\n");
            printf("\n");
        }

    #endif

    /*2nd convolution*/
    /*set up VMM*/
#ifdef ACORE
    reset_VMM();
#else
    free(vmm); // important although only 24 bytes
    vmm = initializeVMM(cnn);
#endif
    int map_size = 3;
    scal = 2;
    column_dex = 0;
    VMM_turns = 0;
    weights_number = 0;
    uint8_t ***outputS2_list = alloc_3darray(4,
                                             14,
                                             14);

    printf("outputS2_list generated!!");

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 14; j++)
            for (int h = 0; h < 14; h++)
                outputS2_list[i][j][h] = *(cnn->S2->y[i][j][h]);
    // outputS2_list[i][j][h] = S2_output[i][j][h];

    freeConvLayer(cnn->C1); // after image generation
    freePoolLayer(cnn->S2); // after image generation

    MatSize input_2nd_size;
    input_2nd_size.columns = (input_size.columns - map_size + 1) / 2;
    input_2nd_size.rows = (input_size.rows - map_size + 1) / 2;
    _CnnSetup(cnn, input_2nd_size, output_size, 2);
    printf("[+] CNN setup finished!\n");
    uint8_t ***weight_array2 = alloc_3darray(scal, IMCcol, IMCrow);
    weights_mapping_Conv(cnn->C3, weight_array2, &weights_number, scal, 2);
    uint8_t ***result_list2 = generate_result_array(scal, 36);
    uint8_t ***VMM_input_array2 = generate_input_array(scal, 36);
    printf("input_2nd_size.columns: %d\n", input_2nd_size.columns);
    printf("input_2nd_size.rows: %d\n", input_2nd_size.rows);
    printf("inputs_mapping_Conv!!!!\n");
    inputs_mapping_Conv(cnn->C3, outputS2_list, VMM_input_array2,
                        &VMM_turns, scal, 2);
    printf("@@@@finish mapping\n");
    for (uint8_t i = 0; i < scal; i++)
    {
        reset_VMM(); // reset VMM to clean VMM, so only need to write non-zero weights
        printf("writing weights!\n");
        printf("\n");
        FeedVMM_weights(weight_array2, i);
        for (int page_image = 0; page_image < VMM_turns; page_image++)
        {
            FeedVMM_image(VMM_input_array2, page_image, i);
#ifdef ACORE
            VMMMACoperation(result_list2, page_image, i); // feedVMM and write VMMmac should be looped by scal right?
#else
            vmm->MACoperation(cnn->C3, VMM_input_array2, result_list2, weight_array2, page_image, i, 2);
#endif
        }
    }
    printf("@@@@@@@@@@@@@Convimage: %d\n", VMM_turns);
    Conv_image(cnn->C3, cnn->S4, result_list2, VMM_turns, weights_number, scal, 2);
    printf("save image!!\n");
    _CnnFF(cnn->C3, cnn->S4);
    #ifndef ACORE

        for (int ch_i = 0; ch_i < 8; ch_i++)
        {
            for (int i = 0; i < 6; i++)
            {
                for (int h = 0; h < 6; h++)
                    printf("%d ", *(cnn->S2->y[ch_i][i][h]));
                printf("\n");
            }
            printf("\n");
        }
    #else

        for (int ch_i = 0; ch_i < 8; ch_i++)
        {
            save_image(6, cnn->S2->y[ch_i]);
            printf("\n");
            printf("\n");
        }

    #endif

    free_3darray(VMM_input_array2, scal, VMM_turns);
    free_3darray(result_list2, scal, VMM_turns);
    free_3darray(weight_array2, scal, IMCcol);
    free_3darray(outputS2_list, 4, 14);

/* fully connected*/
#ifdef ACORE
    reset_VMM();
#else
    free(vmm); // important although only 24 bytes
    vmm = initializeVMM(cnn);
#endif
    map_size = 3;
    scal = 8;
    column_dex = 0;
    VMM_turns = 0;
    weights_number = 0;
    uint8_t ***outputS4_list = alloc_3darray(8,
                                             6,
                                             6);

    printf("outputS2_list generated!!");

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 6; j++)
            for (int h = 0; h < 6; h++)
                outputS4_list[i][j][h] = *(cnn->S4->y[i][j][h]);
    // outputS4_list[i][j][h] = S4_output[i][j][h];

    freeConvLayer(cnn->C3);
    freePoolLayer(cnn->S4);

    MatSize input_3rd_size;
    input_3rd_size.columns = 6 * 6 * 8;
    _CnnSetup(cnn, input_3rd_size, output_size, 3);
    printf("[+] CNN setup finished!\n");
    uint8_t ***weight_array3 = alloc_3darray(1, IMCcol, IMCrow);
    uint8_t ***result_list3 = generate_result_array(1, 8);
    uint8_t ***VMM_input_array3 = generate_input_array(1, 8);
    printf("inputs_mapping_Conv!!!!\n");
    inputs_mapping_FC(cnn->O5, outputS4_list, VMM_input_array3,
                      &VMM_turns, scal, 3);
    free_3darray(outputS4_list, 8, 6);

    printf("@@@@finish mapping\n");

    for (uint8_t i = 0; i < scal; i++)
    {
        weights_mapping_FC(cnn->O5, weight_array3, 3, i);
        reset_VMM(); // reset VMM to clean VMM, so only need to write non-zero weights
        printf("writing weights!\n");
        printf("\n");
        FeedVMM_weights(weight_array3, 0); // updated in the wights mapping
        FeedVMM_image(VMM_input_array3, i, 0);
#ifdef ACORE
        VMMMACoperation(result_list3, i, 0); // feedVMM and write VMMmac should be looped by scal right?
#else
        vmm->MACoperation(cnn->O5, VMM_input_array3, result_list3,
                          weight_array3, i, 0, 3);
#endif
    }

    FC_image(cnn->O5, result_list3, scal, 3);
    printf("save image!!\n");

    for (int i = 0; i < cnn->O5->output_num; i++)
    {
        printf("%d   ", cnn->O5->v[i]);
    }
    printf("\n");

    free_3darray(weight_array3, 1, IMCcol);
    free_3darray(VMM_input_array3, 1, scal);
    free_3darray(result_list3, 1, scal);

/* fully connected*/
#ifdef ACORE
    reset_VMM();
#else
    free(vmm); // important although only 24 bytes
    vmm = initializeVMM(cnn);
#endif

    uint8_t ***outputO5_list = alloc_3darray(1,
                                             1,
                                             32);

    printf("outputS2_list generated!!");

    for (int i = 0; i < 1; i++)
        for (int j = 0; j < 1; j++)
            for (int h = 0; h < 32; h++)
                outputO5_list[i][j][h] = cnn->O5->v[h];

    freeFClayer(cnn->O5);
    scal = 1;
    printf("[+] CNN setup finished!\n");

    uint8_t ***weight_array4 = alloc_3darray(1, IMCcol, IMCrow);
    uint8_t ***result_list4 = generate_result_array(1, 1);
    uint8_t ***VMM_input_array4 = generate_input_array(1, 1);

    inputs_mapping_FC(cnn->O6, outputO5_list, VMM_input_array4,
                      &VMM_turns, scal, 4);

    free_3darray(outputO5_list, 1, 1);

    printf("@@@@finish mapping\n");

    for (uint8_t i = 0; i < scal; i++)
    {
        weights_mapping_FC(cnn->O6, weight_array4, 4, i);
        reset_VMM(); // reset VMM to clean VMM, so only need to write non-zero weights
        printf("writing weights!\n");
        printf("\n");
        FeedVMM_weights(weight_array4, 0); // updated in the wights mapping
        FeedVMM_image(VMM_input_array4, i, 0);
#ifdef ACORE
        VMMMACoperation(result_list4, i, 0); // feedVMM and write VMMmac should be looped by scal right?
#else
        vmm->MACoperation(cnn->O6, VMM_input_array4, result_list4,
                          weight_array4, i, 0, 4);
#endif
    }
    FC_image(cnn->O6, result_list4, scal, 4);
    printf("save image!!\n");
    for (int i = 0; i < cnn->O6->output_num; i++)
    {
        printf("%d   ", cnn->O6->v[i]);
    }
    printf("\n");

    free_3darray(weight_array4, 1, IMCcol);

    free_3darray(VMM_input_array4, 1, scal);
    free_3darray(result_list4, 1, scal);
    freeFClayer(cnn->O6);

#ifndef ACORE
    free(vmm);
#endif
    free(cnn);
#ifdef ACORE
    test_pass();
    test_end();
#endif
}

/*we could compare the Cov result with python model when the pool and RELU are implemented */
