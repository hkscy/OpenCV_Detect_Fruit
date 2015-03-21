/*
============================================================================
Name        : openCV_lab1_01.c
Author      : Christopher Hicks & Raymund Lagua
Version     : 2
Copyright   : GPL
Description : Code that detects fruit at supermarket self-service checkouts.
============================================================================
*/

#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>

#include "NaiveBayes.h"
#include "TrainingDataLinkedList.h"
#include "TrainingData.h"
#include "TestClassification.h"

#ifdef __linux__
	#define snprintf	snprintf
#elif defined _WIN32 || defined _WIN64
	#define snprintf	_snprintf
#else
	#error "Unsupported compile platform"
#endif

typedef struct Box{
	double mean;
	double std_dev;
	double lacunarity;

	struct Box *p_next;
} Box;


// Defines
#define SCREEN_H		768
#define WINDOW_FRILLS	100
#define BUFF			512
#define TRAINING_DATA	"trainingImages.csv"

// Constants
static const char *WINDOW_NAME = "OpenCV Fruit Detection";

IplImage *src,				  /*Original image read from file path @argv[1] */
		 *src_r,			  /*Original image resized to fit on the screen */
		 *src_hsv,			  /*Image in HSV colour space */
		 *fruitMask,		  /*Binary (0 or 255) image result of HSV thresholding */
		 *smoothFruitMask,	  /*Smoothed, binary result of HSV thresholding */
		 *contouredFruitMask, /*Contours of the smoothed, thesholded fruit image */
		 *hsvMeasure,		  /*Measure HSV values by masking src_hsv with smoothFruitMask */
		 *src_cropped,
		 *dst;
CvScalar hsvAvg;

CvCapture *liveImg;

void cvShowAndPause(CvArr *image); /*Show IplImage in window, wait on key press */
int train(char *fileName, CvScalar hsvAvg, double compactness, double texture);		   /*Use the image to build training set */

IplImage* cropSrc(IplImage* src, CvRect rect);
double calcLacunarity(IplImage* cropped_cnv);
Box* add_item(Box *p_head, double mean, double std_dev, double lacunarity);

int main(int argc, char* argv[]) {
	puts("Hello OpenCV!"); /* prints Hello OpenCV! */
	double compactness = 0.0;


	if (argc >= 2 && (src = cvLoadImage(argv[1], 1)) != 0) {

		/* Scale the input image to fit on the screen */
		CvSize scaledSize, origSize = cvGetSize(src);
		float scale = 1.0;
		float h = origSize.height;
		while(h > SCREEN_H - WINDOW_FRILLS) {
			scale -= 0.1;
			h = scale*origSize.height;
		}	/*Once this ends, scale is set */
		scaledSize = cvSize(origSize.width*scale ,origSize.height*scale);

		/*Resize the input image to fir on the screen */
		src_r = cvCreateImage(scaledSize, src->depth, src->nChannels);
		cvResize(src, src_r, CV_INTER_LINEAR);

		cvShowAndPause(src_r); /* Show the original image */
		src_hsv = cvCreateImage(cvGetSize(src_r), src_r->depth, src_r->nChannels);
		cvCvtColor(src_r, src_hsv, CV_RGB2HSV);
		cvShowAndPause(src_hsv); /* Show the HSV image */

		fruitMask = cvCreateImage(cvGetSize(src_hsv), IPL_DEPTH_8U, 1);

		//CV_RGB(r,g,b) returns cvScalar(b,g,r,0)
		//HSV for apples from ImageJ: 15-37, 147-255, 93-255
		//MASSIVE IMPORTANT POINT: OpenCV has Hue values 0-179 rather than 0-255 (i.e. 180 degrees)
		//cvInRangeS(src_hsv, cvScalar(7, 147, 93, 0), cvScalar(179, 255, 255, 0), fruitMask);
		cvInRangeS(src_hsv, cvScalar(18, 127, 93, 0), cvScalar(162, 242, 255, 0), fruitMask);
		cvShowAndPause(fruitMask);

		//Perform median filter to remove outliers and fill holes
		//cvSmooth(src, dst, neighbourhood size, rest not required);
		//neighbourhood size depends on the size of outlier, choose by obverservation for now

		smoothFruitMask = cvCreateImage(cvGetSize(fruitMask), IPL_DEPTH_8U, 1);
		cvSmooth(fruitMask, smoothFruitMask, CV_MEDIAN, 21, 0, 0, 0);
		//cvCopy(hsv_threshold, hsv_filtered, 0);
		cvShowAndPause(smoothFruitMask);

		/* Now the noise has been removed (mostly), find the contours */
		contouredFruitMask = cvCreateImage(cvGetSize(smoothFruitMask), IPL_DEPTH_8U, 1);
		cvCopy(smoothFruitMask, contouredFruitMask, NULL);


		CvMemStorage* storage = cvCreateMemStorage(0); //Structure to store structures such as contours.
		/* Start contour scanning process */
		CvContourScanner ccs = cvStartFindContours(contouredFruitMask,
															  storage,
													sizeof(CvContour),
													 CV_RETR_EXTERNAL,
												 CV_CHAIN_APPROX_NONE,
												 	 	cvPoint(0, 0));

		/* Remove contours which are too small */
		CvSeq* contours = 0;
		int nContours = 0, nRemContours = 0;
		double largestContour = 0.0;
		double area = 0.0;
		while((contours = cvFindNextContour(ccs)) != NULL) {
			double thisContour = cvContourPerimeter( contours );
			printf("%f ", thisContour);
			if( thisContour > largestContour ) {
				largestContour = thisContour;
				area = cvContourArea(contours, CV_WHOLE_SEQ, 0);
			} else {
				cvSubstituteContour( ccs, NULL );
				nRemContours++;
			}
			nContours++;

		}
		contours = cvEndFindContours( &ccs );

		//nContours = cvFindContours(contouredFruitMask, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
		printf("Number of Contours: %d \n", nContours);
		printf("%d contours removed.\n", nRemContours);

		/* Draw contours */
		cvDrawContours(contouredFruitMask, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 1, 1, 8, cvPoint(0, 0));

		compactness = (4 * CV_PI * area) / ( powf(largestContour, 2) );

		printf("Perimeter: %.2f\n", largestContour);
		printf("Area: %.2f\n", area);
		printf("Compactness: %.2f\n", compactness);

		cvShowAndPause(contouredFruitMask);

		/*---------------------------- */
		//Crop an area of the fruit. This is done by creating a bounding box around the contour of the fruit and then editing
		//its points to select a smaller area.

		//Create a bounding box around fruit contour
		CvRect boundingBox = cvBoundingRect(contours, 1);

		//Edit rectangle points to make a smaller rectangle
		boundingBox = cvRect(boundingBox.x + 30, boundingBox.y + 15, boundingBox.width - 60 , boundingBox.height - 30);

		cvRectangle(smoothFruitMask, cvPoint(boundingBox.x, boundingBox.y), cvPoint(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height), cvScalar(128, 128, 128, 0), 1, 4, 0); //show bounding box on fruit
		cvShowAndPause(smoothFruitMask);
		//CvPoint centre = cvPoint(boundingBox.x + (boundingBox.width / 2), boundingBox.y + (boundingBox.height / 2)); //centre point of bounding box + contour

		//This is a function which crops the rectangular area and converts the image to greyscale
		src_cropped = cropSrc(src_r, boundingBox);

		//Calculate lacunarity of region
		double lacunarity = calcLacunarity(src_cropped);
		/*---------------------------- */

		//Next task: Remove outliers (i.e. Noise Reduction -> Remove outliers in ImageJ.
		//Radius: 25 works well, radius determines the area used for calculating the median.
		int row, col, count_black, count_white;
		count_black = 0; count_white = 0;
		uchar pix_value;
		for (row = 0; row < smoothFruitMask->height; row++)
		{
			for (col = 0; col < smoothFruitMask->width; col++)
			{
				pix_value = CV_IMAGE_ELEM(smoothFruitMask, uchar, row, col);
				pix_value == 0 ? count_black++ : count_white++;
			}
		}
		printf("Black pixels: %d, White pixels: %d\n", count_black, count_white);
		double fruitArea = (double)count_white / (count_white + count_black);
		printf("Fruit area: %0.2f%%\n", fruitArea * 100);

		/* Bitwise_and the HSV image with the smoothed binary mask */
		hsvMeasure = cvCreateImage(cvGetSize(src_r), src_r->depth, src_r->nChannels);
		cvCopy(src_hsv, hsvMeasure, smoothFruitMask);
		cvShowAndPause(hsvMeasure);

		/* Take average HSV values from the smoothed mask area of the HSV image data */
		hsvAvg = cvAvg(hsvMeasure, smoothFruitMask);
		printf("Average values, H: %f S: %f V: %f\n", hsvAvg.val[0], hsvAvg.val[1], hsvAvg.val[2]);

		/* Tidy-up */
		cvDestroyWindow(WINDOW_NAME);
		cvReleaseImage(&src_r); 		/*Free source image memory */
		cvReleaseImage(&src_hsv);		/*Free HSV image memory */
		cvReleaseImage(&fruitMask); 	/*Free threshold image memory */
		cvReleaseImage(&smoothFruitMask); /*Free filtered image memory */
	}
	else {
		puts("Image not supplied, program terminated.\n");
		return EXIT_FAILURE;
	}

	/**
	 * If argv[2] is set, it's specifying whether to train or identify using the input image.
	 */
	if( argv[2] != NULL ) {
		if ( (strcmp(argv[2], "t") == 0) ) {	/* Training mode */
			printf("Training mode specified!\n");
			double texture = 1.0;
			return train(argv[1], hsvAvg, compactness, texture);

		} else if( strcmp(argv[2], "i") == 0 ) { /* Identification mode */
			printf("Identification mode specified!\n");
			char *classes[9] = {"braeburn apple",
								"granny smith apple",
								"gala apple",
								"pink lady apple",
								"banana",
								"dragon fruit",
								"orange",
								"mandarin orange",
								"mango"};
			/*Get training data and calculate probabilities */
			TrainingItem *tData = readTrainingData(TRAINING_DATA);
			double texture = 1.0;
			printTList(tData);
			printf("\n");
			Posteriors *pData = calcPosteriors(tData, classes, hsvAvg, compactness, texture);
			//printPList(pData);
			getMostProbableClass(pData);

			freeTList(tData);
		} else {
			printf("Unknown mode %s specified\n", argv[2]);
		}
	}
	return EXIT_SUCCESS;
}

/*
* Displays image in an OpenCV window which remains open until any keyboard key is pressed.
*/
void cvShowAndPause(CvArr *image) {
	cvNamedWindow(WINDOW_NAME, 1);
	cvShowImage(WINDOW_NAME, image);
	cvWaitKey(200);
}

/*
* Use the image to train the system
*/
int train(char *fileName, CvScalar hsvAvg, double compactness, double texture) {
	FILE * testData;
	char * writeBuff;
	//char * fruitName;

	/* Open/Create local training data file */
	if ((testData = fopen(TRAINING_DATA, "a+")) == NULL){ //Open file, create if absent. Append.
		int errsv = errno;
		printf("Failed to create local file for storing training data: %s.\n", strerror(errsv));
		return EXIT_FAILURE;
	}

	/* Check if file already exists in training data */
	// fseek(testData, 0, SEEK_END); /*Move to the end of the file */

	/* Collect data from human operator */
	printf("What type of fruit is this?\n");
	char * fruitName = malloc( BUFF );
	gets(fruitName);

	//printf("How many %ss are there?\n", fruitName);
	//char * numFruit = malloc( BUFF );
	//gets(numFruit);

	int strLen = snprintf(NULL, 0, "%s\t%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",
								   fileName,		// File name image loaded from (*)
								   fruitName,		// User input fruit name
								   // numFruit,
								   hsvAvg.val[HUE], 	// H (does this need to be scaled back)
								   hsvAvg.val[SATURATION], 	// S
								   hsvAvg.val[VALUE],	// V
								   compactness,
								   texture);

	writeBuff = malloc(strLen + 1);
	snprintf(writeBuff, BUFF, "%s\t%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",
			   	   	   	   	   fileName,		// File name image loaded from (*)
							   fruitName,		// User input fruit name
							   // numFruit,
							   hsvAvg.val[0], 	// H (does this need to be scaled back)
							   hsvAvg.val[1], 	// S
							   hsvAvg.val[2],	// V
							   compactness,
							   texture);
	fwrite(writeBuff, strLen, 1, testData);

	fclose(testData);
	free(writeBuff);
	free(fruitName);
	// free(numFruit);
	return EXIT_SUCCESS;
}

/*
 * Crop src to calculate texture
 */
IplImage* cropSrc(IplImage* src, CvRect rect)
{
	//Create storage for an image of the size of the ROI rectangle
	IplImage* cropped = cvCreateImage(cvSize(rect.width, rect.height), src->depth, src->nChannels);

	//Set the region of interest in src image as rect
	cvSetImageROI(src, rect);

	//Copy the ROI image to the cropped image container
	cvCopy(src, cropped, NULL);

	//Reset the ROI of src to default
	cvResetImageROI(src);

	cvShowAndPause(cropped);

	//Create  storage for the colour converted image
	IplImage* cropped_cnv = cvCreateImage(cvGetSize(cropped), IPL_DEPTH_8U, 1);

	//Convert from RGB to GREYSCALE
	cvCvtColor(cropped, cropped_cnv, CV_RGB2GRAY);

	printf("Cropped rows, cols: %d, %d\n", cropped->height, cropped->width);

	cvShowAndPause(cropped_cnv);

	return cropped_cnv;
}


/*
* Calculate lacunarity measure of texture.
*/
double calcLacunarity(IplImage* cropped_cnv)
{

	// The empty list is represented by a pointer to NULL.
	Box *p_head = NULL;

	printf("rows: %d, cols: %d\n", cropped_cnv->height, cropped_cnv->width); //Print size of cropped image

	int boxSize = 30; //This is the length of the side of the box

	CvPoint box = cvPoint(0, 0); //
	CvRect roi;

	IplImage* temp = cvCreateImage(cvSize(boxSize, boxSize), cropped_cnv->depth, cropped_cnv->nChannels);

	int iteration_y = (cropped_cnv->height / boxSize);//number of boxes that could fit in the y direction
	int iteration_x = (cropped_cnv->width / boxSize);//number of boxes that could fit in the x direction

	int noBoxes = iteration_y * iteration_x;//number of boxes that could fit in the cropped image

	printf("iteration: %d\n", iteration_y);
	printf("iteration: %d\n", iteration_x);

	double value; //stores the values needed to calculate the mean and variance. NEED TO BE ARRAY
	double mean_; //stores the mean value of a box
	double variance_; //stores the variance of a box
	double std_dev_; //stores the standard deviation of a box
	double lacunarity_ = 0.0;

	for (int y = 0; y < iteration_y * boxSize; y = y + boxSize)
	{
		for (int x = 0; x < iteration_x * boxSize; x = x + boxSize)
		{
			value = 0;
			mean_ = 0.0;
			variance_ = 0.0;
			std_dev_ = 0.0;

			//box = cvPoint(x, y);

			/* THIS WILL SHOW DRAW THE RECTANGLES TO SHOW THE BOXES LAID OVER THE IMAGE
			cvRectangle(cropped_cnv, box, cvPoint(x + boxSize, y + boxSize), cvScalar(255, 255, 255, 0), 1, 4, 0);
			cvShowAndPause(cropped_cnv);*/

			cvSetImageROI(cropped_cnv, cvRect(x, y, boxSize, boxSize)); //set ROI to particular rectangle
			cvCopy(cropped_cnv, temp, NULL); //copy this to a temporary storage
			cvResetImageROI(cropped_cnv); //reset ROI
			cvShowAndPause(temp);

			//Calculate the mean value of pixels in the ROI
			for (int row = 0; row < temp->height;row++)
			{
				for (int col = 0; col < temp->width; col++)
				{
					value = value + CV_IMAGE_ELEM(temp, uchar, row, col);
					//printf("value: %f\n", value);
				}
			}

			mean_ = value / (temp->height*temp->width);
			printf("mean: %f\n", mean_);
			value = 0.0;

			//Calculate the variance of pixels in the ROI
			for (int row = 0; row < temp->height; row++)
			{
				for (int col = 0; col < temp->width; col++)
				{
					value = value + powf((CV_IMAGE_ELEM(temp, uchar, row, col) - mean_), 2) ;
					//printf("value: %f\n", value);
				}
			}

			variance_ = value / (temp->height*temp->width);
			printf("variance: %f\n", variance_);

			//Calculate standar deviation
			std_dev_ = sqrt(variance_);
			printf("std_dev: %f\n", std_dev_);

			lacunarity_ = powf((std_dev_ / mean_), 2);
			printf("lacunarity: %f\n", lacunarity_);
			//cvShowAndPause(cropped_cnv);

			// Add s at the beginning of the list
			p_head = add_item(p_head, mean_, std_dev_, lacunarity_);

		}
	}

	printf("The list is:\n");
	Box *p_current_item = p_head;
	while (p_current_item) {    // Loop while the current pointer is not NULL
		//printf("test");
		if (p_current_item->mean && p_current_item->std_dev && p_current_item->lacunarity) {
			printf("mean: %f\nstd_dev: %f\nlacunarity: %f\n", p_current_item->mean, p_current_item->std_dev, p_current_item->lacunarity);
		}
		else {
			printf("No data\n");
		}
		// Advance the current pointer to the next item in the list.
		p_current_item = p_current_item->p_next;
	}

	printf("end\n");
	return 0;
}


Box* add_item(Box *p_head, double mean, double std_dev, double lacunarity){

	Box* p_new_item = malloc(sizeof(Box));

	p_new_item -> p_next = p_head;
	p_new_item -> mean = mean;
	p_new_item -> std_dev = std_dev;
	p_new_item -> lacunarity = lacunarity;

	return p_new_item;
}
