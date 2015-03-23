/*
 * TrainingData.h
 *
 *  Created on: 16 Mar 2015
 *      Author: Chris Hicks & Ray Lagua
 */
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

#define LINE_BUFF 512

#define FILEN_COL	0
#define SCLASS_COL	1
#define ACLASS_COL	2
#define H_COL		3
#define S_COL		4
#define V_COL		5
#define C_COL		6
#define T_COL		7

/**
 * Given the path to the training data, which must be CSV separated and contain
 * fields for all the members of a TrainingItem.
 *
 * Returns a single linked list of training data
 *
 * WARNING: Memory is allocated for fileNames and classes.
 */
TrainingItem *readTrainingData(char *fileName) {

	FILE * testData;
	TrainingItem *tData = NULL;

	/* Open training data file */
	if ((testData = fopen(fileName, "rb")) == NULL){ //Open file, fp at beginning
		int errsv = errno;
		printf("Failed to open training data from %s with error: %s\n", fileName, strerror(errsv));
		return NULL;//EXIT_FAILURE;
	}

	/* Determine total size of training data & allocate memory for it. */
	fseek(testData, 0, SEEK_END);
	size_t bytesTotal = (size_t)(ftell( testData ));
	printf("Training data file size: %lu\n", bytesTotal);
	char *rBuff = malloc( bytesTotal );

	/* Read training data into memory */
	fseek(testData, 0, SEEK_SET);
	fread(rBuff, bytesTotal, 1, testData);
	// printf("%s", rBuff); - Print training data in entirety.

	/* Check for errors reading data into memory */
	if (ferror(testData)) {
		printf("WARNING: Test data file not properly read.");
		return NULL;
	}

	/* Split training data into lines/entries */
	char *data = rBuff;
	char *p_line  = strchr(data, '\n');

	while (p_line != NULL)
	{
		/* Split training data lines into CSV columns */
		uint8_t columnN = 0;
		char * fName = malloc( LINE_BUFF );
		char * class = malloc( LINE_BUFF );
		double h = 0, s = 0, v = 0, c = 0, t = 0;
		*p_line++ = '\0';
	    // printf("parsing: %s\n", data); - debug, prints out one line of training data
	    char *p_val = strtok(data, "\t");
	    while (p_val != NULL)
	    {
	    	if(FILEN_COL == columnN) { /*File Name */
	    		strcpy(fName, p_val);
	    	} else if (SCLASS_COL == columnN) { /*Class */
	    		//strcpy(class, p_val);
	    	} else if (ACLASS_COL == columnN) { /*Class */
	    		strcpy(class, p_val);
	    	} else if (H_COL == columnN) { /*H */
	    		h = strtod(p_val, NULL);
	    	} else if (S_COL == columnN) { /*S */
	    		s = strtod(p_val, NULL);
	    	} else if (V_COL == columnN) { /*V */
	    		v = strtod(p_val, NULL);
	    	} else if (C_COL == columnN) { /*Compactness */
	    		c = strtod(p_val, NULL);
	    	} else if (T_COL == columnN) { /*Texture */
	    		t = strtod(p_val, NULL);
	    	} else {
	    		printf("Unknown training data in file: %s\n", p_val);
	    	}
	        p_val = strtok(NULL, "\t");
	        columnN++;
	    }
	    tData = addTItem(tData, class, h, s, v, c, t);	/*Add data to list */
	    data = p_line;
	    p_line = strchr(data, '\n');
	    free(fName);
	}

	/* Flush and close training data file, free buffers */
	if(fclose(testData) != 0) {
		int errsv = errno;
		printf("Failed to properly close training data file at %s with error: %s\n", fileName, strerror(errsv));
		return NULL; //EXIT_FAILURE;
	}
	free(rBuff);

	return tData;
}



#endif /* TRAININGDATA_H_ */
