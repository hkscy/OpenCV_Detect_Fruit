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

// Defines
#define SCREEN_H		800
#define WINDOW_FRILLS	100
#define BUFF			512
#define TRAINING_DATA	"trainingImages.csv"

// Constants
static const char *WINDOW_NAME = "OpenCV Fruit Detection";

IplImage *src,				  /*Original image read from file path @argv[1] */
		 *src_r,			  /*Original image resized to fit on the screen */
		 *src_hsv,			  /*Image in HSV colour space */
		 *fruitMask,		  /*Binary (0 or 255) image result of HSV thresholding */
		 *erodeMask,
		 *smoothMask,	  	/*Smoothed, binary result of HSV thresholding */
		 *contouredMask, 	/*Contours of the smoothed, thesholded fruit image */
		 *hsvMeasure,		  /*Measure HSV values by masking src_hsv with smoothFruitMask */
		 *textureMeasureSquare,
		 *dst;
CvScalar hsvAvg;

HSV_Range classes[NCLASSES] = {
	{"braeburn apple", 0, 28, 170, 180, 144, 255, 100, 255},
	{"granny smith apple", 24, 70, 24, 70, 79, 255, 40, 160},
	{"gala apple", 165, 180, 0, 9, 90, 245, 40, 255},
	{"banana", 15, 38, 15, 38, 134, 255, 120, 255},
	{"dragon fruit", 0, 52, 17, 51, 109, 255, 30, 148},
	{"mandarin orange", 0, 20, 0, 18, 139, 255, 90, 255},
	{"mango", 19, 61, 19, 51, 108, 255, 27, 150},
};

void cvShowAndPause(CvArr *image); /*Show IplImage in window, wait on key press */
int train(char *imageFileName, CvScalar hsvAvg, double compactness, double texture, char *selectedClass, char *actualClass);	/*Use the image to build training set */

IplImage* cropSrc(IplImage* src, CvRect rect);
double calcLacunarity(IplImage* cropped_cnv);
CvRect cropFruit(IplImage* hsv_filtered, CvRect boundingBox);

int main(int argc, char* argv[]) {
	puts("Hello OpenCV!"); /* prints Hello OpenCV! */
	double area = 0.0;
	double fruitArea = 0.0;
	double compactness = 0.0;
	double texture = 1.0;
	char *selectedClass = malloc(BUFF);
	int8_t classN = -1;

	if (argc >= 3 && (src = cvLoadImage(argv[1], 1)) != 0) {

		/* Third argument specifies mode */
		if( argv[2] != NULL ) {
			if ( (strcmp(argv[2], "t") == 0) ) {	/* Training mode */
				printf("Training mode specified!\n");

			} else if( strcmp(argv[2], "i") == 0 ) { /* Identification mode */
				printf("Identification mode specified!\n");

			} else {
				printf("Unknown mode %s specified\n", argv[2]);
				return EXIT_FAILURE;
			}
		}

		printf("\nWhich fruit have you put on the scales?\n\nPlease choose from:\n");
		int count = 0;
		for(; count < NCLASSES; count++) {
			printf("%d: %s\n", count, classes[count].class);
		}
		char *userInput = malloc( BUFF );
		strcpy(selectedClass, "");

		while(selectedClass[0] == '\0') {
			gets(userInput);
			//strcpy(userInput, "6"); /*Debug */
			for(count = 0; count < NCLASSES; count++) {
				 if(strcmp(classes[count].class, userInput) == 0) {
					 strcpy(selectedClass, classes[count].class);
					 classN = count;
					 break;
				 }
			}
			if(selectedClass[0] == '\0' && (strlen(userInput) == 1)) {
				int userInputInt = atoi(userInput);
				if(userInputInt >= 0 && userInputInt < NCLASSES) {
					strcpy(selectedClass, classes[userInputInt].class);
					classN = userInputInt;
					break;
				}
			}
			if(selectedClass[0] == '\0') {
				printf("Please make a valid selection.\n");
			}
		}
		free(userInput);

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
		cvCvtColor(src_r, src_hsv, CV_BGR2HSV);
		cvShowAndPause(src_hsv); /* Show the HSV image */

		fruitMask = cvCreateImage(cvGetSize(src_hsv), IPL_DEPTH_8U, 1);
		IplImage *hsvA = cvCreateImage(cvGetSize(src_hsv), IPL_DEPTH_8U, 1);
		IplImage *hsvB = cvCreateImage(cvGetSize(src_hsv), IPL_DEPTH_8U, 1);
		//CV_RGB(r,g,b) returns cvScalar(b,g,r,0)
		//HSV for apples from ImageJ: 15-37, 147-255, 93-255
		//MASSIVE IMPORTANT POINT: OpenCV has Hue values 0-179 rather than 0-255 (i.e. 180 degrees)
		//cvInRangeS(src_hsv, cvScalar(7, 147, 93, 0), cvScalar(179, 255, 255, 0), fruitMask);
		cvInRangeS(src_hsv,
				   cvScalar(classes[classN].h1, classes[classN].s1, classes[classN].v1, 0),
				   cvScalar(classes[classN].h2, classes[classN].s2, classes[classN].v2, 0),
				   hsvA);
		cvInRangeS(src_hsv,
				   cvScalar(classes[classN].h3, classes[classN].s1, classes[classN].v1, 0),
				   cvScalar(classes[classN].h4, classes[classN].s2, classes[classN].v2, 0),
				   hsvB);
		cvOr(hsvA, hsvB, fruitMask, NULL);
		printf("HSV= %u %u %u\n", classes[classN].h1, classes[classN].s1, classes[classN].v1);

		//cvInRangeS(src_hsv, cvScalar(0, 90, 60, 0), cvScalar(14, 225, 205, 0), hsvB);
		//cvOr(hsvA, hsvB, fruitMask, NULL);
		cvShowAndPause(fruitMask);

		//Perform median filter to remove outliers and fill holes
		//cvSmooth(src, dst, neighbourhood size, rest not required);
		//neighbourhood size depends on the size of outlier, choose by obverservation for now

		erodeMask = cvCreateImage(cvGetSize(fruitMask), IPL_DEPTH_8U, 1);
		smoothMask = cvCreateImage(cvGetSize(fruitMask), IPL_DEPTH_8U, 1);

		/* Erode fruit to separate touching fruits */
		IplConvKernel* element;
		//IplConvKernel* cvCreateStructuringElementEx(int cols, int rows, int anchor_x, int anchor_y, int shape, int* values=NULL );
		element = cvCreateStructuringElementEx(4, 4, 1, 1, 1, NULL);
		cvErode(fruitMask, erodeMask, element, 3);

		cvSmooth(erodeMask, smoothMask, CV_MEDIAN, 11, 0, 0, 0);
		//cvCopy(erodeMask, smoothFruitMask, 0);
		cvShowAndPause(smoothMask);

		/* Now the noise has been removed (mostly), find the contours */
		contouredMask = cvCreateImage(cvGetSize(smoothMask), IPL_DEPTH_8U, 1);
		cvCopy(smoothMask, contouredMask, NULL);


		CvMemStorage* storage = cvCreateMemStorage(0); //Structure to store structures such as contours.
		/* Start contour scanning process */
		CvContourScanner ccs = cvStartFindContours(contouredMask,
															  storage,
													sizeof(CvContour),
													 CV_RETR_EXTERNAL,
												 CV_CHAIN_APPROX_NONE,
												 	 	cvPoint(0, 0));

		/* Remove contours which are too small */
		CvSeq* contours = 0;
		int nContours = 0, nRemContours = 0;
		double largestContour = 0.0;
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
		if(nContours > 0) {
			/* Draw contours (filled) */
			cvDrawContours(contouredMask, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 1, CV_FILLED, 8, cvPoint(0, 0));

			compactness = (4 * CV_PI * area) / ( powf(largestContour, 2) );
			printf("Perimeter: %.2f\n", largestContour);
			printf("Area: %.2f\n", area);
			printf("Compactness: %.2f\n", compactness);

			cvShowAndPause(contouredMask);
			//cvCopy(contouredMask, smoothMask, contouredMask);

			/*---------------------------- */
			//Create a bounding box around fruit contour
			CvRect boundingBox = cvBoundingRect(contours, 1);

			//Find the area where a box can be fitted inside the fruit
			CvRect crop = cropFruit(contouredMask, boundingBox);

			//This is a function which crops the rectangular area and converts the image to greyscale
			textureMeasureSquare = cropSrc(src, crop);

			//Calculate lacunarity of region
			double lacunarity = calcLacunarity(textureMeasureSquare);
			printf("Lacunarity: %.4f \n", lacunarity);
			cvShowAndPause(contouredMask);
		}
		/*---------------------------- */

		//Next task: Remove outliers (i.e. Noise Reduction -> Remove outliers in ImageJ.
		//Radius: 25 works well, radius determines the area used for calculating the median.
		int row, col, count_black, count_white;
		count_black = 0; count_white = 0;
		uchar pix_value;
		for (row = 0; row < contouredMask->height; row++)
		{
			for (col = 0; col < contouredMask->width; col++)
			{
				pix_value = CV_IMAGE_ELEM(contouredMask, uchar, row, col);
				/* Force everything other than 255 to be 0 */
				pix_value == 255 ? (CV_IMAGE_ELEM(contouredMask, uchar, row, col)=255) : (CV_IMAGE_ELEM(contouredMask, uchar, row, col)=0);
				pix_value == 0 ? count_black++ : count_white++;
			}
		}
		printf("Black pixels: %d, White pixels: %d\n", count_black, count_white);
		fruitArea = ((double)count_white / (count_white + count_black))*100;
		printf("Fruit area: %0.3f%%\n", fruitArea);

		/* Bitwise_and the HSV image with the smoothed binary mask */
		hsvMeasure = cvCreateImage(cvGetSize(src_r), src_r->depth, src_r->nChannels);
		cvCopy(src_hsv, hsvMeasure, contouredMask);
		cvShowAndPause(hsvMeasure);
		cvShowAndPause(contouredMask);

		/* Take average HSV values from the smoothed mask area of the HSV image data */
		hsvAvg = cvAvg(hsvMeasure, contouredMask);
		printf("Average values, H: %f S: %f V: %f\n", hsvAvg.val[0], hsvAvg.val[1], hsvAvg.val[2]);

		/* Tidy-up */
		cvDestroyWindow(WINDOW_NAME);
		cvReleaseImage(&src_r); 		/*Free source image memory */
		cvReleaseImage(&src_hsv);		/*Free HSV image memory */
		cvReleaseImage(&fruitMask); 	/*Free threshold image memory */
		cvReleaseImage(&smoothMask); /*Free filtered image memory */
	}
	else if(argc == 2) {
		puts("Mode not specified, program terminated. (Use 't' to train the system, 'i' to test it)\n");
	}
	else {
		puts("Image or mode not specified, program terminated.\n");
		return EXIT_FAILURE;
	}

	/**
	 * If argv[2] is set, it's specifying whether to train or identify using the input image.
	 */
	if( argv[2] != NULL ) {
		if ( (strcmp(argv[2], "t") == 0) ) {	/* Training mode */
			if(hsvAvg.val[0] > 0 && hsvAvg.val[1] > 0 && hsvAvg.val[2] > 0 && (fruitArea > 0.2)) {
				char * actualClass = malloc( BUFF );
				int count = 0;

				printf("\nWhich fruit is shown in the image?\n\nPlease choose from:\n");
				for(; count < NCLASSES; count++) {
					printf("%d: %s\n", count, classes[count].class);
				}
				char *userInput = malloc( BUFF );
				strcpy(actualClass, "");

				while(actualClass[0] == '\0') {
					gets(userInput);

					for(count = 0; count < NCLASSES; count++) {
						 if(strcmp(classes[count].class, userInput) == 0) {
							 strcpy(actualClass, classes[count].class);
							 break;
						 }
					}
					if(actualClass[0] == '\0' && (strlen(userInput) == 1)) {
						int userInputInt = atoi(userInput);
						if(userInputInt >= 0 && userInputInt < NCLASSES) {
							strcpy(actualClass, classes[userInputInt].class);
							break;
						}
					}
					if(selectedClass[0] == '\0') {
						printf("Please make a valid selection.\n");
					}
				}
				free(userInput);
				train(argv[1], hsvAvg, compactness, texture, selectedClass, actualClass);
				free(actualClass);
			} else {
				printf("Average HSV values are all 0, not writing to file.\n");
			}
		} else if( strcmp(argv[2], "i") == 0 ) { /* Identification mode */
			if(hsvAvg.val[0] > 0 && hsvAvg.val[1] > 0 && hsvAvg.val[2] > 0 && (fruitArea > 0.2)) {
				printf("fruitarea: %f\n", fruitArea);
				char * fName = malloc( BUFF );
				strcpy(fName, selectedClass);
				strcat(fName, ".csv");

				/*Get training data and calculate probabilities */
				TrainingItem *tData = readTrainingData(fName);
				//printTList(tData);
				printf("\n");
				Posteriors *pData = calcPosteriors(tData, classes, hsvAvg, compactness, texture);
				//printPList(pData);

				char * detected = getMostProbableClass(pData);
				if(strcmp(detected, selectedClass) == 0) {
					printf("Please add the item to your basket.\n\n");
				} else if(detected[0] != '\0') {
					printf("Please wait for assistance, wrong item detected.\n\n");
					printf("You've most probably put %s in your basket by mistake.\n", detected);
				} else {
					printf("Please wait for assistance, wrong item detected.\n\n");
					printf("The contents of your basket is unidentifiable.\n");
				}

				freeTList(tData);
				free(fName);
			} else {
				printf("Please wait for assistance, wrong item detected.\n\n");
				printf("The contents of your basket is unidentifiable.\n");
			}
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
int train(char *imageFileName, CvScalar hsvAvg, double compactness, double texture, char *selectedClass, char* actualClass) {
	FILE * testData;
	char * writeBuff;
	char * fName = malloc( BUFF );
	strcpy(fName, selectedClass);
	strcat(fName, ".csv");

	/* Open/Create local training data file */
	if ((testData = fopen(fName, "a+")) == NULL){ //Open file, create if absent. Append.
		int errsv = errno;
		printf("Failed to create local file for storing training data: %s.\n", strerror(errsv));
		return EXIT_FAILURE;
	}

	int strLen = snprintf(NULL, 0, "%s\t%s\t%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",
								   imageFileName,			// File name image loaded from (*)
								   selectedClass,					// User input fruit name
								   actualClass,
								   hsvAvg.val[HUE], 		// H (does this need to be scaled back)
								   hsvAvg.val[SATURATION], 	// S
								   hsvAvg.val[VALUE],		// V
								   compactness,
								   texture);

	writeBuff = malloc( strLen + 1 );
	snprintf(writeBuff, BUFF, "%s\t%s\t%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",
			   	   	   	   	   imageFileName,		// File name image loaded from (*)
							   selectedClass,		// User input fruit name
							   actualClass,
							   hsvAvg.val[0], 		// H (does this need to be scaled back)
							   hsvAvg.val[1], 		// S
							   hsvAvg.val[2],		// V
							   compactness,
							   texture);
	fwrite(writeBuff, strLen, 1, testData);

	fclose(testData);
	free(fName);
	free(writeBuff);
	return EXIT_SUCCESS;
}

/*
 * Crop src to calculate texture
 */
IplImage* cropSrc(IplImage* src, CvRect rect)
{
	//Create storage for an image of the size of the ROI rectangle
	IplImage* cropped = cvCreateImage(cvSize(rect.width, rect.height), src_r->depth, src_r->nChannels);

	//Set the region of interest in src image as rect
	cvSetImageROI(src_r, rect);

	//Copy the ROI image to the cropped image container
	cvCopy(src_r, cropped, NULL);

	//Reset the ROI of src to default
	cvResetImageROI(src_r);

	cvShowAndPause(cropped);

	//Create  storage for the colour converted image
	IplImage* cropped_cnv = cvCreateImage(cvGetSize(cropped), IPL_DEPTH_8U, 1);

	//Convert from RGB to GREYSCALE
	cvCvtColor(cropped, cropped_cnv, CV_RGB2GRAY);

	printf("Cropped rows, cols: %d, %d\n", cropped->height, cropped->width);

	cvShowAndPause(cropped_cnv);

	return cropped_cnv;
}

CvRect cropFruit(IplImage* hsv_filtered, CvRect boundingBox)
{
	int box_size = 50;

	/***DEBUG***/
	/*cvRectangle(hsv_filtered, cvPoint(boundingBox.x, boundingBox.y), cvPoint(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height), cvScalar(128, 128, 128, 0), 1, 4, 0); //show bounding box on fruit
	cvShowAndPause(src_contours);
	cvShowAndPause(hsv_filtered);*/

	//Determine centre point of bounding box
	CvPoint centre = cvPoint(boundingBox.x + (boundingBox.width / 2), boundingBox.y + (boundingBox.height / 2)); //centre point of bounding box

	//cvCircle(hsv_filtered, centre, 2, cvScalar(128, 128, 128, 0), 1, CV_FILLED, 0); //DEBUG: Draw centre point

	//Create a box with around the centre point
	CvRect crop = cvRect(centre.x - (box_size / 2), centre.y - (box_size / 2), box_size, box_size);

	/*cvRectangle(hsv_filtered,
				cvPoint(centre.x - 25, centre.y - 25),
				cvPoint(centre.x + 25, centre.y + 25),
				cvScalar(128, 128, 128, 0),
				1,
				1,
				0); //DEBUG: Draw Rectangle on image*/

	/*Check if all pixels are white in centre box*/

	int flag_black = 0; //Flag to check whether any black pixels are detected inside box

	for (int row = crop.y; (row < crop.y + 50) && (flag_black != 1); row++)
	{
		for (int col = crop.x; (col < crop.x + 50) && (flag_black != 1); col++)
		{
			if (CV_IMAGE_ELEM(hsv_filtered, uchar, row, col) == 0)
			{
				flag_black = 1; //set flag
			}
		}
	}

	/***If a black pixel was found in the initial box, start from top of central part of the image***/
	/***and iterate from top to bottom until the box contains no black pixels***/

	if (flag_black == 1)
	{
		flag_black = 0; //reset flag

		for (int iterate = 0; iterate < hsv_filtered->height - 50; iterate++)
		{
			for (int row = iterate; (row < 50 + iterate) && (flag_black != 1); row++)
			{
				for (int col = crop.x; (col < crop.x + 50) && (flag_black != 1); col++)
				{
					if (CV_IMAGE_ELEM(hsv_filtered, uchar, row, col) == 0)
					{
						flag_black = 1;
					}
				}
			}

			if (flag_black == 1)
			{
				flag_black = 0;
			}
			else //If no black pixels found break the loop
			{
				crop = cvRect(crop.x, iterate, 50, 50);
				break;
			}
		}

		//cvRectangle(hsv_filtered, cvPoint(crop.x, crop.y), cvPoint(crop.x + 50, crop.y + 50), cvScalar(128, 128, 128, 0), 1, 1, 0);// DEBUG: Draw rectangle on image
		//cvShowAndPause(hsv_filtered);

		/***Once first box with all white pixels has been found, iterate further and find middle point***/

		int counter = 0; // Counter to see how much further the box will contain all white pixels
		flag_black = 0;


		for (int iterate = crop.y; iterate < hsv_filtered->height - 50; iterate++)
		{

			for (int row = iterate; (row < iterate + 50) && (flag_black != 1); row++)
			{
				for (int col = crop.x; (col < crop.x + 50) && (flag_black != 1); col++)
				{
					if (CV_IMAGE_ELEM(hsv_filtered, uchar, row, col) == 0)
					{
						flag_black = 1;
					}
				}
			}

			if (flag_black == 1)
			{
				break;
			}
			else
			{
				counter++;
			}

		}

		/***Divide counter by 2 and add to rectangle location from previous process to find***/
		/***the center point where the box can fit. If counter is odd, counter-- and divide***/

		if ((counter % 2 == 0) && counter != 0 && counter != 1) //counter is even
		{
			crop = cvRect(crop.x, crop.y + (counter / 2), 50, 50);
			cvRectangle(hsv_filtered, cvPoint(crop.x, crop.y), cvPoint(crop.x + 50, crop.y + 50), cvScalar(128, 128, 128, 0), 1, 1, 0);//DEBUG: Draw rectangle
			cvShowAndPause(hsv_filtered);
		}
		else if (counter % 2 == 1 && counter != 0 && counter != 1) //counter is odd
		{
			counter--;
			crop = cvRect(crop.x, crop.y + (counter / 2), 50, 50);
			cvRectangle(hsv_filtered, cvPoint(crop.x, crop.y), cvPoint(crop.x + 50, crop.y + 50), cvScalar(128, 128, 128, 0), 1, 1, 0);//DEBUG: Draw rectangle
			cvShowAndPause(hsv_filtered);
		} else {
			printf("\n\nNeither condition is met \n\n");
		}

	}

	return crop;

}

/*
* Calculate lacunarity of fruit
*/

double calcLacunarity(IplImage* cropped_cnv)
{

	printf("rows: %d, cols: %d\n", cropped_cnv->height, cropped_cnv->width); //Print size of cropped image

	int boxSize = 10; //This is the length of the side of the box

	CvPoint box = cvPoint(0, 0); //
	CvRect roi;

	IplImage* temp = cvCreateImage(cvSize(boxSize, boxSize), cropped_cnv->depth, cropped_cnv->nChannels);

	int iteration_y = (cropped_cnv->height / boxSize);//number of boxes that could fit in the y direction
	int iteration_x = (cropped_cnv->width / boxSize);//number of boxes that could fit in the x direction

	int noBoxes = iteration_y * iteration_x;//number of boxes that could fit in the cropped image

	//printf("iteration: %d\n", iteration_y);
	//printf("iteration: %d\n", iteration_x);

	double value; //stores the values needed to calculate the mean and variance. NEED TO BE ARRAY
	double mean; //stores the mean value of a box
	double variance; //stores the variance of a box
	double std_dev; //stores the standard deviation of a box
	double lacunarity = 0.0;

	for (int y = 0; y < iteration_y * boxSize; y = y + boxSize)
	{
		for (int x = 0; x < iteration_x * boxSize; x = x + boxSize)
		{
			value = 0;
			mean = 0.0;
			variance = 0.0;
			std_dev = 0.0;

			//box = cvPoint(x, y);

			/* THIS WILL SHOW DRAW THE RECTANGLES TO SHOW THE BOXES LAID OVER THE IMAGE
			cvRectangle(cropped_cnv, box, cvPoint(x + boxSize, y + boxSize), cvScalar(255, 255, 255, 0), 1, 4, 0);
			cvShowAndPause(cropped_cnv);*/

			cvSetImageROI(cropped_cnv, cvRect(x, y, boxSize, boxSize)); //set ROI to particular rectangle
			cvCopy(cropped_cnv, temp, NULL); //copy this to a temporary storage
			cvResetImageROI(cropped_cnv); //reset ROI
			//cvShowAndPause(temp);

			//Calculate the mean value of pixels in the ROI
			for (int row = 0; row < temp->height; row++)
			{
				for (int col = 0; col < temp->width; col++)
				{
					value = value + CV_IMAGE_ELEM(temp, uchar, row, col);
					//printf("value: %f\n", value);
				}
			}

			mean = value / (temp->height*temp->width);
			//printf("mean: %f\n", mean);


			//Calculate the variance of pixels in the ROI
			for (int row = 0; row < temp->height; row++)
			{
				for (int col = 0; col < temp->width; col++)
				{
					value = value + powf((CV_IMAGE_ELEM(temp, uchar, row, col) - mean), 2);
					//printf("value: %f\n", value);
				}
			}

			variance = value / (temp->height*temp->width);
			//printf("variance: %f\n", variance);

			//Calculate standard deviation
			std_dev = sqrt(variance);
			//printf("std_dev: %f\n", std_dev);

			lacunarity = lacunarity + powf((std_dev / mean), 2);

		}
	}

	lacunarity /= noBoxes; //calculate average lacunarity

	return lacunarity;

}
