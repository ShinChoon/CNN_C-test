#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <stdint.h>
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

void FeedVMM_weights(uint8_t ***weight_array)
{
    int page_weight = 0;
    int index_weight = 0;
    uint8_t feed = 0; // do not change this please
    for (int i = 2 * IMCrow; i < (IMCrow - 2) * IMCrow; i++)
    {
        page_weight = (int)(i / IMCrow) - 2;
        index_weight = i % IMCrow;
        feed = weight_array[0][page_weight][index_weight];
#ifdef ACORE
        write(i, feed);
#endif
    }
}

void FeedVMM_image(uint8_t ***VMMarray, uint8_t pagenumber, uint8_t scal)
{
    uint8_t feed = 0;
    for (uint8_t i = 0; i < IMCrow; i++)
    {
        feed = VMMarray[scal][pagenumber][i];
#ifdef ACORE
        write(i, feed);
#endif
    }
}
void VMMMACoperation(uint8_t ***result_list, int pagenumber, int scal)
{
    uint8_t result = 0;
    for (int h = 0; h < IMCcol; h++)
    {
        result = (uint8_t)read(h);
        result_list[scal][pagenumber][h] = result;
        // printf("%d ", (int)(result_list[0][pagenumber][h]));
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
    _CnnSetup(cnn, input_size, output_size, 1);
    printf("[+] CNN setup finished!\n");

    // /*load weights from csv file, not needed anymore*/
    _ImportCnn(cnn, 1);
    printf("[+] Import CNN finished!\n");

    /*map weigths and iamge as in IMC shape*/

    int VMM_turns = 0;
    int weights_number = 0;
    int turn_number = 0;
    int column_dex = 0;
    /*Weights can be reused right????*/
    MnistImage *input_list[] = {&(test_images->image_point[0])};

    /*below are for the 1st convolution*/
    /*weights*/
    int scal = 1; // scal for the 1st conv, as 2 in the 2nd convolution
    uint8_t ***weight_array = alloc_3darray(scal, IMCcol, IMCrow);
    weights_mapping(cnn->C1, weight_array, &weights_number, scal);
    /*remember to release the memory and produce cnn again after 1st MAC*/
    uint8_t ***VMM_input_array = generate_input_array(scal, 112);
    uint8_t ***result_list = generate_result_array(scal, 112);

#ifdef ACORE
    reset_VMM();
#else
    VMM *vmm = initializeVMM(cnn);
#endif

    printf("writing weights!\n");
    FeedVMM_weights(weight_array);
    printf("\n");

    for (uint8_t i = 0; i < scal; i++)
        /*foor loop by 4 here*/
        {
            /*image making values*/
            printf("inputs_mapping!!!!\n");
            inputs_mapping(cnn->C1, input_list, VMM_input_array, &VMM_turns, scal);
            printf("@@@@finish mapping\n");

            for (int page_image = 0; page_image < VMM_turns; page_image++)
            {
                // printf("writing image\n");
                FeedVMM_image(VMM_input_array, page_image, i);
#ifdef ACORE
                printf("reading\n");
                VMMMACoperation(result_list, page_image, i); // feedVMM and write VMMmac should be looped by scal right?

#else
                vmm->MACoperation(VMM_input_array, result_list, weight_array, page_image, scal);
#endif
            }
            printf("@@@@@@@@@@@@@Convimage: %d\n", VMM_turns);
            Conv_image(cnn->C1, cnn->S2, result_list, VMM_turns, weights_number, scal, &column_dex);
        }

    free_3darray(VMM_input_array, scal, 112);
    free_3darray(result_list, scal, 112);

    printf("save image!!\n");
    _CnnFF(cnn->C1, cnn->S2);
    for (int ch_i = 0; ch_i < 4; ch_i++)
    {
        save_image(14, cnn->S2->y[ch_i]);
        printf("\n");
        printf("\n");
    }
    freeConvLayer(cnn->C1);

    // /*2nd convolution*/
    // int map_size = 3;
    // scal = 2;
    // column_dex = 0;
    // VMM_turns = 0;
    // MatSize input_2nd_size;
    // uint8_t ***VMM_input_array2 = generate_input_array(scal, 28);
    // uint8_t ***result_list2 = generate_result_array(scal, 28);
    // input_2nd_size.columns = (input_size.columns - map_size + 1) / 2;
    // input_2nd_size.rows = (input_size.rows - map_size + 1) / 2;

    // ImageArray outputS2 = Output_image(input_2nd_size.columns,
    //                         input_2nd_size.columns,
    //                         cnn->S2->y, cnn->S2->output_channels);

    // MnistImage *outputS2_list[] = {&(outputS2->image_point[0]),
    //                                &(outputS2->image_point[1]),
    //                                &(outputS2->image_point[2]),
    //                                &(outputS2->image_point[3])};

    // printf("input_2nd_size.columns: %d\n", input_2nd_size.columns);
    // printf("input_2nd_size.rows: %d\n", input_2nd_size.rows);

    // freePoolLayer(cnn->S2);

    // _CnnSetup(cnn, input_size, output_size, 2);
    // printf("[+] CNN setup finished!\n");

    // // /*load weights from csv file, not needed anymore*/
    // _ImportCnn(cnn, 2);
    // printf("[+] Import CNN finished!\n");

//     uint8_t ***weight_array2 = alloc_3darray(scal, IMCcol, IMCrow);

//     weights_mapping(cnn->C3, weight_array2, &weights_number, scal);

//     printf("writing weights!\n");
//     FeedVMM_weights(weight_array2);
//     printf("\n");

//     for (uint8_t i = 0; i < scal; i++)
//         /*foor loop by 4 here*/
//         for (turn_number = 0; turn_number < 7; turn_number++)//7*28 = 196 
//         {
//             printf("inputs_mapping!!!!\n");
//             inputs_mapping(cnn->C3, outputS2_list, VMM_input_array2,
//                            &VMM_turns, scal, turn_number, i);

//             printf("@@@@finish mapping\n");
//             for (int page_image = 0; page_image < 28; page_image++)
//             {
//                 FeedVMM_image(VMM_input_array2, page_image, i);

//                 #ifdef ACORE
//                                 printf("reading\n");
//                                 VMMMACoperation(result_list2, page_image); // feedVMM and write VMMmac should be looped by scal right?

// #else
//                 vmm->MACoperation(VMM_input_array2, result_list2, weight_array2, page_image, scal);
//                 #endif
//             }
//             printf("@@@@@@@@@@@@@Convimage: %d\n", VMM_turns);
//             Conv_image(cnn->C3, 28, result_list2, VMM_turns, weights_number, scal, &column_dex);
//         }

//     free_3darray(VMM_input_array2, scal, 28);
//     free_3darray(result_list2, scal, 28);

//     printf("save image!!\n");
//     _CnnFF(cnn->C3, cnn->S4);
//     for (int ch_i = 0; ch_i < 4; ch_i++)
//     {
//         save_image(6, cnn->S4->y[ch_i]);
//         printf("\n");
//         printf("\n");
//     }
    // freeConvLayer(cnn->C3);

    // char *filename = (char *)malloc(sizeof(char) * 13);
    // float ***weight2_array = weights_mapping(cnn->C3, &weights_number,2);
    // float *bias2_array = bias_mapping(cnn->C3, &bias_number
    // /*debug: create pgm files, later use convert in terminal to create png*/
    // /*save data as image*/
    // sprintf(filename, "image_%d.pgm", 1);
    // save_image(cnn->C1->input_height,
    //         test_images->image_point[0].image_data,
    //         filename);
    // for (int i = 0; i < cnn->C1->output_channels; i++)
    // {
    //     save_image(cnn->S2->input_height, cnn->C1->v[i]);
    // }
    // for (int i = 0; i < cnn->S2->output_channels; i++)
    // {
    //     save_image(cnn->S2->input_height / 2, cnn->S2->y[i]);
    // }

    // for (int i = 0; i < cnn->C3->output_channels; i++)
    // {
    //     save_image(cnn->S4->input_height, cnn->C3->v[i]);
    // }
    // for (int i = 0; i < cnn->S4->output_channels; i++)
    // {
    //     save_image(cnn->S4->input_height / 2, cnn->S4->y[i]);
    // }

#ifdef ACORE
    test_pass();
    test_end();
#endif
            }

/*we could compare the Cov result with python model when the pool and RELU are implemented */
