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

/**
 * Given the class of fruit, iterate the list of training data and
 * calculate the mean of the attribute selected.
 * 0: Hue
 * 1: Saturation
 * 2: Value
 * 3: Compactness
 */
float calcHSVC_Mean(TrainingItem * p_head, char *fruitName, uint8_t attr) {
	float mean = 0.0;
	int divisor = 0;
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
float calcHSVC_SD(TrainingItem * p_head, char * fruitName, uint8_t attr) {

	int divisor = 0;
	float avg = calcHSVC_Mean(p_head, fruitName, attr);
	float sumOfSqDiff = 0.0; /* Sum of squared differences (x-avg)^2 */
	float sd = 0.0;

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

	return sd;
}

/**
 * P(Class_fruitName|h)
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
float calcHSVC_PDF(TrainingItem * p_head, char * fruitName, float h) {
	return 0.0;
}

#endif /*NATIVEBAYES_H_*/
