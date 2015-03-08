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

#include "NativeBayes.h"
#include "TrainingDataLinkedList.h"
#include "TestClassification.h"

#ifdef __linux__
	#define snprintf	snprintf
#elif defined _WIN32 || defined _WIN64
	#define snprintf	_snprintf
#else
	#error "Unsupported compile platform"
#endif


// Defines
#define SCREEN_H		768
#define WINDOW_FRILLS	100
#define BUFF			1024
#define TRAINING_DATA	"trainingImages.csv"

// Constants
static const char *WINDOW_NAME = "OpenCV Fruit Detection";

IplImage *src,				/*Original image read from file path @argv[1] */
		 *src_r,			/*Original image resized to fit on the screen */
		 *src_hsv,			/*Image in HSV colour space */
		 *fruitMask,	/*Binary (0 or 255) image result of HSV thresholding */
		 *hsv_masked,
		 *hsv_filtered,
		 *src_contours,
		 *dst;
CvScalar hsvAvg;

void cvShowAndPause(CvArr *image); /*Show IplImage in window, wait on key press */
int train(char *fileName, CvScalar hsvAvg, float compactness);		   /*Use the image to build training set */

int main(int argc, char* argv[]) {
	puts("Hello OpenCV!"); /* prints Hello OpenCV! */
	float compactness = 0.0;

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
		cvInRangeS(src_hsv, cvScalar(7, 147, 93, 0), cvScalar(179, 255, 255, 0), fruitMask);
		cvShowAndPause(fruitMask);

		//Perform median filter to remove outliers and fill holes
		//cvSmooth(src, dst, neighbourhood size, rest not required);
		//neighbourhood size depends on the size of outlier, choose by obverservation for now

		hsv_filtered = cvCreateImage(cvGetSize(fruitMask), IPL_DEPTH_8U, 1);
		cvSmooth(fruitMask, hsv_filtered, CV_MEDIAN, 21, 0, 0, 0);
		//cvCopy(hsv_threshold, hsv_filtered, 0);
		cvShowAndPause(hsv_filtered);

		//Find contours using freeman's chain code algorithm
		//We only require the external contours thus any "holes" in our thresholded object

		//Structure to store structures such as CvSeq
		CvMemStorage* storage = cvCreateMemStorage(0);

		//Structure to store contour points 
		CvSeq* contours = 0;

		//Find contours in image, desirably, we would only have one distinct object after the thresholding and smoothing operations
		//Only find external contours and store all contour points
		int nc = cvFindContours(hsv_filtered, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
		printf("Number of Contours: %d \n", nc);

		//Find largest contour
		//*THIS MIGHT BE NEEDED, DEPENDS HOW GOOD WE CAN THRESHOLD OUR IMAGES*

		//Draw contours on image
		src_contours = cvCreateImage(cvGetSize(hsv_filtered), IPL_DEPTH_8U, 1);
		cvSet(src_contours, CV_RGB(255, 255, 255), 0);
		cvDrawContours(hsv_filtered, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 1, 1, 8, cvPoint(0, 0));


		//Calculate area and mean
		double perimeter = cvArcLength(contours, CV_WHOLE_SEQ, 0);
		double area = cvContourArea(contours, CV_WHOLE_SEQ, 0);
		compactness = (4 * CV_PI * area) / ( powf(perimeter, 2) );

		printf("Perimeter: %.2f\n", perimeter);
		printf("Area: %.2f\n", area);
		printf("Compactness: %.2f\n", compactness);

		cvShowAndPause(hsv_filtered);

		//Next task: Remove outliers (i.e. Noise Reduction -> Remove outliers om ImageJ.
		//Radius: 25 works well, radius determines the area used for calculating the median.
		int row, col, count_black, count_white;
		count_black = 0; count_white = 0;
		uchar pix_value;
		for (row = 0; row < fruitMask->height; row++)
		{
			for (col = 0; col < fruitMask->width; col++)
			{
				pix_value = CV_IMAGE_ELEM(fruitMask, uchar, row, col);
				pix_value == 0 ? count_black++ : count_white++;
			}
		}
		printf("Black pixels: %d, White pixels: %d\n", count_black, count_white);
		float fruitArea = (float)count_white / (count_white + count_black);
		printf("Fruit area: %0.2f%%\n", fruitArea * 100);

		//Bitwise_and the HSV image with the threshold bitmap, should mask just the fruit
		hsv_masked = cvCreateImage(cvGetSize(src_r), src_r->depth, src_r->nChannels);
		//cvAnd(src_hsv, hsv_threshold, hsv_masked, NULL);
		cvCopy(src_hsv, hsv_masked, fruitMask);
		cvShowAndPause(hsv_masked);

		hsvAvg = cvAvg(hsv_masked, fruitMask);
		printf("Average values, H: %f S: %f V: %f\n", hsvAvg.val[0], hsvAvg.val[1], hsvAvg.val[2]);

		/*Save these to a file for building test data */

		/*Match to nearest neighbour for identifying fruit */

		/* Tidy-up */
		cvDestroyWindow(WINDOW_NAME);
		cvReleaseImage(&src_r); 		/*Free source image memory */
		cvReleaseImage(&src_hsv);	/*Free HSV image memory */
		cvReleaseImage(&fruitMask); /*Free threshold image memory */
		cvReleaseImage(&hsv_filtered); /*Free filtered image memory */
		cvReleaseImage(&src_contours); /*Free contours image memory */
	}
	else {
		puts("Image not supplied, program terminated.\n");
		return EXIT_FAILURE;
	}

	/**
	 * If argv[2] is set, it's specifying whether to train or identify using the input image.
	 */
	if( argv[2] != NULL ) {
		if ( (strcmp(argv[2], "t") == 0) ) {
			printf("Training mode specified!\n");
			return train(argv[1], hsvAvg, compactness);
		} else if( strcmp(argv[2], "i") == 0 ) {
			printf("Identification mode specified!\n");
			testBayes();
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
	cvWaitKey(0);
}

/*
* Use the image to train the system
*/
int train(char *fileName, CvScalar hsvAvg, float compactness) {
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

	int strLen = snprintf(NULL, 0, "%s\t%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",
								   fileName,		// File name image loaded from (*)
								   fruitName,		// User input fruit name
								   // numFruit,
								   hsvAvg.val[0], 	// H (does this need to be scaled back)
								   hsvAvg.val[1], 	// S
								   hsvAvg.val[2],	// V
								   compactness);

	writeBuff = malloc(strLen + 1);
	snprintf(writeBuff, BUFF, "%s\t%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",
			   	   	   	   	   fileName,		// File name image loaded from (*)
							   fruitName,		// User input fruit name
							   // numFruit,
							   hsvAvg.val[0], 	// H (does this need to be scaled back)
							   hsvAvg.val[1], 	// S
							   hsvAvg.val[2],	// V
							   compactness);
	fwrite(writeBuff, strLen, 1, testData);

	fclose(testData);
	free(writeBuff);
	free(fruitName);
	// free(numFruit);
	return EXIT_SUCCESS;
}
