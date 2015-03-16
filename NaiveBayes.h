/**
 *	Header containing the methods necessary for performing native Bayes classification.
 *	08/03/15
 *
 *	Christopher Hicks & Raymund Lagua
 */
#include <math.h>
#include "TrainingDataLinkedList.h"

#ifndef NATIVEBAYES_H_
#define NATIVEBAYES_H_

#define PI 3.14159

/**
 * Given the class of fruit, iterate the list of training data and
 * calculate the mean of the attribute selected.
 * 0: Hue
 * 1: Saturation
 * 2: Value
 * 3: Compactness
 */
double calcHSVC_Mean(TrainingItem * p_head, char *fruitName, uint8_t attr) {
	double mean = 0.0;
	double divisor = 0.0;
	TrainingItem *p_current_item = p_head;
	while (p_current_item) {    /* Loop while the current pointer is not NULL. */
		if((strcmp(p_current_item->fruitName, fruitName) == 0) &&
				   p_current_item->h &&	/*If valid item */
				   p_current_item->s &&
				   p_current_item->v) {

			if(HUE == attr) {
				mean += p_current_item->h;
			} else if(SATURATION == attr) {
				mean += p_current_item->s;
			} else if(VALUE == attr) {
				mean += p_current_item->v;
			} else if(COMPACTNESS == attr) {
				mean += p_current_item->c;
			}
			/* Add error checking here on attribute type */
			divisor++;
		}
		/* Advance the current pointer to the next item in the list */
		p_current_item = p_current_item->p_next;
	}
	if(divisor > 0.0) {
		mean = mean/divisor;
	}
	//printf("mean: %f for %u\n", mean, attr);
	return mean;
}

/**
 * Given the class of fruit, iterate the list of training data and
 * calculate the Standard Deviation for the attribute specified.
 *
 * "For a finite set of numbers, the standard deviation is found by
 *  taking the square root of the average of the squared differences
 *  of the values from their average value" - Wikipedia
 *
 * 0: Hue
 * 1: Saturation
 * 2: Value
 * 3: Compactness
 */
double calcHSVC_SD(TrainingItem * p_head, char * fruitName, uint8_t attr) {

	double divisor = 0.0;
	double avg = calcHSVC_Mean(p_head, fruitName, attr);
	double sumOfSqDiff = 0.0; /* Sum of squared differences (x-avg)^2 */
	double sd = 0.0;

	TrainingItem *p_current_item = p_head;
	while (p_current_item) {    /* Loop while the current pointer is not NULL. */
		if((strcmp(p_current_item->fruitName, fruitName) == 0) &&
				   p_current_item->h &&	/*If valid item */
				   p_current_item->s &&
				   p_current_item->v &&
				   p_current_item->c) {

			/* Calculate difference between average and this value, squared */
			if(HUE == attr) {
				sumOfSqDiff +=  pow(p_current_item->h-avg, 2);
			} else if(SATURATION == attr) {
				sumOfSqDiff +=  pow(p_current_item->s-avg, 2);
			} else if(VALUE == attr) {
				sumOfSqDiff +=  pow(p_current_item->v-avg, 2);
			} else if(COMPACTNESS == attr) {
				sumOfSqDiff +=  pow(p_current_item->c-avg, 2);
			}
			divisor++;
		}
		/* Advance the current pointer to the next item in the list */
		p_current_item = p_current_item->p_next;
	}

	sd = sqrt( sumOfSqDiff/divisor );
	//printf("sd: %f for %u\n", sd, attr);

	return sd;
}

/**
 * p(C{fruitName}|attr=val)
 * Calculate the probability of a given attribute belonging to a given
 * class (fruitName).
 * 1/sqrt(2*pi*SD^2)*exp((x-mean)^2/(2*SD^2))
 *
 * attributes:
 * 0: Hue
 * 1: Saturation
 * 2: Value
 * 3: Compactness
 */
double calcHSVC_PDF(TrainingItem * p_head, char * fruitName, uint8_t attr, double val) {

	double pdf = 0.0;
	double mean = calcHSVC_Mean(p_head, fruitName, attr);
	double sd = calcHSVC_SD(p_head, fruitName, attr);;
	double a = (1/(sd*sqrt(2*CV_PI)));
	double b = exp( ((-1)*(pow((val-mean),2)) / (2*pow(sd,2))) );

	pdf = a * b;

	return pdf;
}

/**
 *	Calculates the posterior probability density for the specified class.
 *	Equal prior probability is assumed to P(class) = 1/n
 *
 *	posterior(class) = prior(class)*p(h|class)*p(s|class)*p(v|class)*p(compactness|class)
 */
double calcPosterior(TrainingItem *tDataHead, char *class, CvScalar sampleHSV, double sampleC) {

	double post = 0.0;
	double pH = calcHSVC_PDF(tDataHead, class, HUE, sampleHSV.val[HUE]);
	double pS = calcHSVC_PDF(tDataHead, class, SATURATION, sampleHSV.val[SATURATION]);
	double pV = calcHSVC_PDF(tDataHead, class, VALUE, sampleHSV.val[VALUE]);
	double pC = calcHSVC_PDF(tDataHead, class, COMPACTNESS, sampleC);
	/*printf("P(hue|%s) %f, P(sat|%s) %f, P(val|%s) %f, P(c|%s) %f\n", class, pH,
																	 class, pS,
																	 class, pV,
																     class, pC); */
	post = pH*pS*pV*pC;
	printf("posterior(%s) = %0.80f\n", class, post);

	return post;
}

/**
 * Get the list of posterior probabilities from the TrainingData list
 *
 * *classes[] - Array of class names for which to calculate the posteriors
 *  hsvSample - CvScalar containing the H, S and V values from the sample to be identified.
 *  cSample   - Measure of compactness from the sample to be identified.
 */
Posteriors *calcPosteriors(TrainingItem *tData, char *classes[], CvScalar hsvSample, double cSample) {

	Posteriors *post = NULL;
	int class = 0;
	for(class = 0; class < sizeof(classes); class++) {
		double pProb = calcPosterior(tData, classes[class], hsvSample, cSample);
		post = addPosterior(post, classes[class], pProb);
	}

	return post;
}


#endif /*NATIVEBAYES_H_*/
