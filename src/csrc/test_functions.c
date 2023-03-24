#include <stdlib.h>  //random
#include <stdio.h>
#include <time.h>    //random seed
#include "cnn.h" 

// Test Mnist data. Save the images to local.
void TestMnist()
{
  LabelArray test_label = ReadLabels("mnist/t10k-labels-idx1-ubyte");
  ImageArray test_image = ReadImages("mnist/t10k-images-idx3-ubyte");
	SaveImage(test_image, "mnist/test_images"); //Save images
}

//Test Mat functions.
void TestMat()
{
	MatSize srcSize = {6,6};
	MatSize map_size = {4,4};

  //time to random seed.
	srand((unsigned)time(NULL)); 
  
	//define and random src mat
	uint8_t** src = (uint8_t**)malloc(srcSize.rows * sizeof(uint8_t*));
	for(int i=0; i<srcSize.rows; i++)
	{
		src[i] = (uint8_t*)malloc(srcSize.columns * sizeof(uint8_t));

		for(int j=0; j<srcSize.columns; j++)
		{
			//Generate uint8_t from [-1:1]
			src[i][j] = (((uint8_t)rand() / (uint8_t)RAND_MAX)-0.5) * 2; 
		}
	}

  //define and random map mat
	uint8_t** map = (uint8_t**)malloc(map_size.rows * sizeof(uint8_t*));
	for(int i=0; i<map_size.rows; i++)
	{
		map[i] = (uint8_t*)malloc(map_size.columns * sizeof(uint8_t));

		for(int j=0; j<map_size.columns; j++)
		{
			//Generate uint8_t from [-1:1]
			map[i][j] = (((uint8_t)rand() / (uint8_t)RAND_MAX)-0.5) * 2; 
		}
	}

	MatSize cov1size={srcSize.columns+map_size.columns-1,srcSize.rows+map_size.rows-1};
	uint8_t** cov1=MatCov(map,map_size,src,srcSize,FULL);
	//MatSize cov2size={srcSize.columns,srcSize.rows};
	//uint8_t** cov2=MatCov(map,map_size,src,srcSize,SAME);
	MatSize cov3size={srcSize.columns-(map_size.columns-1),srcSize.rows-(map_size.rows-1)};
	uint8_t** cov3=MatCov(map,map_size,src,srcSize,VALID);

}

void TestMat1()
{
	int i,j;
	MatSize srcSize={12,12};
	MatSize map_size={5,5};
	uint8_t** src=(uint8_t**)malloc(srcSize.rows*sizeof(uint8_t*));
	for(i=0;i<srcSize.rows;i++){
		src[i]=(uint8_t*)malloc(srcSize.columns*sizeof(uint8_t));
		for(j=0;j<srcSize.columns;j++){
			src[i][j]=(((uint8_t)rand()/(uint8_t)RAND_MAX)-0.5)*2; 
		}
	}
	uint8_t** map1=(uint8_t**)malloc(map_size.rows*sizeof(uint8_t*));
	for(i=0;i<map_size.rows;i++){
		map1[i]=(uint8_t*)malloc(map_size.columns*sizeof(uint8_t));
		for(j=0;j<map_size.columns;j++){
			map1[i][j]=(((uint8_t)rand()/(uint8_t)RAND_MAX)-0.5)*2; 
		}
	}
	uint8_t** map2=(uint8_t**)malloc(map_size.rows*sizeof(uint8_t*));
	for(i=0;i<map_size.rows;i++){
		map2[i]=(uint8_t*)malloc(map_size.columns*sizeof(uint8_t));
		for(j=0;j<map_size.columns;j++){
			map2[i][j]=(((uint8_t)rand()/(uint8_t)RAND_MAX)-0.5)*2; 
		}
	}
	uint8_t** map3=(uint8_t**)malloc(map_size.rows*sizeof(uint8_t*));
	for(i=0;i<map_size.rows;i++){
		map3[i]=(uint8_t*)malloc(map_size.columns*sizeof(uint8_t));
		for(j=0;j<map_size.columns;j++){
			map3[i][j]=(((uint8_t)rand()/(uint8_t)RAND_MAX)-0.5)*2; 
		}
	}

	uint8_t** cov1=MatCov(map1,map_size,src,srcSize,VALID);
	uint8_t** cov2=MatCov(map2,map_size,src,srcSize,VALID);
	MatSize covsize={srcSize.columns-(map_size.columns-1),srcSize.rows-(map_size.rows-1)};
	uint8_t** cov3=MatCov(map3,map_size,src,srcSize,VALID);
	MatAdd(cov1,cov1,covsize,cov2,covsize);
	MatAdd(cov1,cov1,covsize,cov3,covsize);



}

void test_cnn()
{

	LabelArray test_label = ReadLabels("mnist/train-labels-idx1-ubyte");
	ImageArray test_image = ReadImages("mnist/train-images-idx3-ubyte");

	MatSize input_size = {test_image->image_point[0].number_of_columns, \
	                  test_image->image_point[0].number_of_rows};
  
	//Output size
	int output_size = test_label->label_point[0].label_length; 

	Cnn* cnn=(Cnn*)malloc(sizeof(Cnn));

  //Setup the CNN
	CnnSetup(cnn,input_size,output_size);

  //Train the CNN
	TrainOptions opts;
	opts.numepochs=1;  //train epochs
	opts.alpha=1;
	int num_train=5000; //Train number
	CnnTrain(cnn,test_image,test_label,opts,num_train);
  
	//output
	FILE *file_point = NULL;
	file_point = fopen("output/cnn_layer.ma","wb");
	if(file_point == NULL)
		printf("[-] Write file failed! <output/cnn_layer.ma>\n");

	fwrite(cnn->L, sizeof(uint8_t), num_train, file_point);
	fclose(file_point);
}