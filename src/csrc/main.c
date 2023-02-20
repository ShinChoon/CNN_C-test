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

void FeedVMM_weights(uint8_t ***weight_array, int VMM_turns, int Scaling)
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

void FeedVMM_image(uint8_t ***VMMarray, int pagenumber, int Scaling)
{
    uint8_t feed = 0;
    for (int sc = 0; sc < Scaling; sc++)
    {
        for (int i = 0; i < IMCrow; i++)
        {
            feed = VMMarray[0][pagenumber][i];
#ifdef ACORE
            write(i, feed);
#endif
        }
    }
}
void VMMMACoperation(uint8_t ***result_list, int pagenumber, int Scaling)
{
    uint8_t result = 0;
    for (int h = 0; h < IMCcol; h++)
    {
        result = (uint8_t)read(h);
        result_list[0][pagenumber][h] = result;
        // printf("%d ", (int)(result_list[0][pagenumber][h]));
    }
    // printf("\n");
}

void free_data(int ***data, size_t xlen, size_t ylen)
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

uint8_t ***alloc_3darray(size_t xlen, size_t ylen, size_t zlen)
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

uint8_t **alloc_2darray(size_t xlen, size_t ylen)
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

uint8_t *alloc_1darray(size_t xlen)
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

    // /*load weights from csv file, not needed anymore*/
    // _ImportCnn(cnn);
    // printf("[+] Import CNN finished!\n");

    /*map weigths and iamge as in IMC shape*/

    int VMM_turns = 0;
    int weights_number = 0;
    int bias_number = 0;
    int page_image = 0;
    int result = 0;
    int h = 0;
    int temp = 0;

#ifdef ACORE
    // uint8_t ***result_list = generate_result_array();
    uint8_t ***result_list = alloc_3darray(1, 28, IMCcol);
#endif

    uint8_t ***weight_array = weights_mapping(cnn->C1, &weights_number, 1);
    uint8_t *bias_array = bias_mapping(cnn->C1);
    int turn_number = 0;
    int column_dex = 0;

    MnistImage *input_list[] = {&(test_images->image_point[0])};
    uint8_t ***VMM_input_array = alloc_3darray(1, 28, IMCrow);
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

#ifndef DEBUG
        for (page_image = 0; page_image < 28; page_image++)
        {
            printf("writing image\n");
            FeedVMM_image(VMM_input_array, page_image, 1);
#ifdef ACORE
            printf("reading\n");
            VMMMACoperation(result_list, page_image, 1);
#endif
        }
#endif

#ifndef ACORE
        uint8_t ***result_list = vmm->MACoperation(VMM_input_array, weight_array, 28, 1);
#endif

        printf("@@@@@@@@@@@@@Convimage: %d\n", VMM_turns);
        Conv_image(cnn->C1, 28, result_list, VMM_turns, weights_number, 1, &column_dex);
    }

    printf("save image!!\n");
    _CnnFF(cnn->C1, cnn->S2, bias_array);
    for (int ch_i = 0; ch_i < cnn->S2->output_channels; ch_i++)
    {
        save_image(14, cnn->S2->y[ch_i]);
        printf("\n");
        printf("\n");
    }

#ifdef ACORE
    test_pass();
    test_end();
#endif
}

/*we could compare the Cov result with python model when the pool and RELU are implemented */
