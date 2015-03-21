/*
 * TestClassification.h
 *
 *  Created on: 8 Mar 2015
 *      Author: Christopher Hicks & Raymund Lagua
 *
 * Tests the functionality of NativeBayes.h with some static data
 */

#include "NaiveBayes.h"
#include "TrainingDataLinkedList.h"


void testBayes(CvScalar hsvVal, double compactness) {
	/*	Build TrainingDataLinkedList using the following sample data.
	 *
	 *	fruit	H		S		V		Compactness
		banana	95.19	177.48	184.28	0.22
		banana	94.81	168.97	172.53	0.22
		banana	95.54	173.27	174.66	0.30
		banana	94.26	200.05	164.95	0.23
		banana	94.45	195.85	168.41	0.34
		banana	97.24	202.45	159.22	0.35
		orange	107.31	232.85	170.92	0.78
		orange	106.95	233.78	199.56	0.74
		orange	107.59	224.42	154.52	0.74
		granny	85.85	202.16	119.99	0.75
		granny	84.69	204.09	123.55	0.85
		granny	83.84	200.62	114.50	0.85
		granny	87.78	197.40	128.16	0.83
		granny	85.33	204.23	128.71	0.85
		granny	87.20	220.39	117.27	0.86
		granny	87.04	216.43	112.86	0.85
		granny	87.86	212.36	115.22	0.88
	 */

	const int numData = 17;
	char *fruitNames[numData];
	double h[numData]; double s[numData]; double v[numData]; double c[numData];
	fruitNames[0] = "banana";	  h[0] = 95.19;  s[0] = 177.48;  v[0] = 184.28;  c[0] = 0.22;
	fruitNames[1] = "banana"; 	  h[1] = 94.81;  s[1] = 168.97;  v[1] = 172.53;	 c[1] = 0.22;
	fruitNames[2] = "banana"; 	  h[2] = 95.54;  s[2] = 173.27;  v[2] = 174.66;  c[2] = 0.30;
	fruitNames[3] = "banana"; 	  h[3] = 94.26;  s[3] = 200.05;  v[3] = 164.66;	 c[3] = 0.23;
	fruitNames[4] = "banana"; 	  h[4] = 94.45;  s[4] = 195.85;  v[4] = 168.41;  c[4] = 0.34;
	fruitNames[5] = "banana";	  h[5] = 97.24;  s[5] = 202.45;  v[5] = 159.22;  c[5] = 0.35;
	fruitNames[6] = "orange";	  h[6] = 107.31; s[6] = 232.85;  v[6] = 170.92;	 c[6] = 0.78;
	fruitNames[7] = "orange";	  h[7] = 106.95; s[7] = 233.78;  v[7] = 199.56;	 c[7] = 0.74;
	fruitNames[8] = "orange";	  h[8] = 107.59; s[8] = 224.42;  v[8] = 154.52;	 c[8] = 0.74;
	fruitNames[9] = "granny";	  h[9] = 85.85;  s[9] = 202.16;  v[9] = 119.99;	 c[9] = 0.75;
	fruitNames[10] = "granny";    h[10] = 84.69; s[10] = 204.09; v[10] = 123.55; c[10] = 0.85;
	fruitNames[11] = "granny";    h[11] = 83.84; s[11] = 200.62; v[11] = 114.50; c[11] = 0.85;
	fruitNames[12] = "granny";	  h[12] = 87.78; s[12] = 197.40; v[12] = 128.16; c[12] = 0.83;
	fruitNames[13] = "granny";    h[13] = 85.33; s[13] = 204.23; v[13] = 128.71; c[13] = 0.85;
	fruitNames[14] = "granny";    h[14] = 87.20; s[14] = 220.39; v[14] = 117.27; c[14] = 0.86;
	fruitNames[15] = "granny";	  h[15] = 87.04; s[15] = 216.43; v[15] = 112.85; c[15] = 0.85;
	fruitNames[16] = "granny";	  h[16] = 87.86; s[16] = 212.36; v[16] = 115.22;  c[16] = 0.88;
	double texture = 1.0;

	int i;
	TrainingItem *p_head = NULL; 	/* 'Initialise' empty list */
	for (i=0; i<numData; i++) {
		/* Build list of training data */
	    p_head = addTItem(p_head, fruitNames[i], h[i], s[i], v[i], c[i], 0.0);
	}
	printTList(p_head);

	double pGranny = calcPosterior(p_head, "granny", hsvVal, compactness, texture);
	double pBanana = calcPosterior(p_head, "banana", hsvVal, compactness, texture);
	double pOrange = calcPosterior(p_head, "orange", hsvVal, compactness, texture);

	Posteriors *post = NULL;
	post = addPosterior(post, "granny", pGranny);
	post = addPosterior(post, "banana", pBanana);
	post = addPosterior(post, "orange", pOrange);

	printPList(post);

	getMostProbableClass(post);
	//printf("Fruit identified as: %s\n", idFruit);

	freeTList(p_head);

}
