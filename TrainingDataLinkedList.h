/*
 * TrainingDataLinkedList.h
 *
 *  Created on: 8 Mar 2015
 *      Author: Christopher Hicks & Raymund Lagua
 *
 * Defines elements of training data which can be linked together.
 */
#include <stdlib.h>

#ifndef LINKEDLIST_H_
#define LINKEDLIST_H_

#define HUE 		0
#define SATURATION 	1
#define	VALUE		2
#define COMPACTNESS 3

/**
 * Defines one element of training data, and a pointer to the next.
 */
typedef struct TrainingItem {
	char 	*fruitName;
	double  h; /* Hue */
	double  s; /* Saturation */
	double  v; /* Value */
	double  c; /* Compactness */
  struct TrainingItem *p_next;
} TrainingItem;

/**
 * Defines an element of posterior probability, and a pointer to the next.
 */
typedef struct Posteriors {
	char 	*class;
	double	posteriorP;
	struct Posteriors *p_next;
} Posteriors;

/**
 * Adds new item to list of training data, returns pointer to new list element
 * as head of list. (i.e. FILO buffer)
 */
TrainingItem* addTItem(TrainingItem *p_head, char *fruitName, double h, double s, double v, double c) {
  // printf("Adding item: %s\t%0.2f\t%0.2f\t%0.2f", fruitName, h, s , v);
  TrainingItem *p_new_item = malloc(sizeof(TrainingItem));
  p_new_item->p_next = p_head;			/* Set pointer to previous head */
  p_new_item->fruitName = fruitName;    /* Set data pointers */
  p_new_item->h = h;
  p_new_item->s = s;
  p_new_item->v = v;
  p_new_item->c = c;

  return p_new_item;
}

Posteriors* addPosterior(Posteriors *p_head, char *class, double posteriorP) {
  Posteriors *p_new_item = malloc( sizeof(Posteriors) );
  p_new_item->p_next = p_head;	/* Set pointer to previous head */
  p_new_item->class = class;    /* Set data pointers */
  p_new_item->posteriorP = posteriorP;

  return p_new_item;
}

/*
 * Traverses the TrainingData and prints off the attributes of each list item.
 * Useful for debugging and testing.
 */
void printTList(TrainingItem *p_head) {

	TrainingItem *p_current_item = p_head;
	while (p_current_item) {    /* Loop while the current pointer is not NULL. */
		if(p_current_item->fruitName && p_current_item->h &&
										p_current_item->s &&
										p_current_item->v &&
										p_current_item->c) {
			printf("%s\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n", p_current_item->fruitName,
											  		   p_current_item->h,
													   p_current_item->s,
													   p_current_item->v,
													   p_current_item->c);
		} else {
			printf("No data.\n");
		}
		/* Advance the current pointer to the next item in the list */
		p_current_item = p_current_item->p_next;
	}
}

void printPList(Posteriors *p_head) {
	Posteriors *p_current_item = p_head;
		while (p_current_item) {    /* Loop while the current pointer is not NULL. */
			if(p_current_item->class && p_current_item->posteriorP) {
				printf("%s\t%0.200f\n", p_current_item->class, p_current_item->posteriorP);
			}
			/* Advance the current pointer to the next item in the list */
			p_current_item = p_current_item->p_next;
		}
}

int getPListLen(Posteriors *p_head) {

	int count = 0;
	Posteriors *p_current = p_head;
	while(p_current) {
		count++;
		p_current = p_current->p_next;
	}
	return count;
}

/*
 * Free all of the TrainingItem elements, and the structure itself.
 */
int freeTList(TrainingItem *p_head)	{

	printf("Freeing Training Data\n: ");
	TrainingItem *p_current_item = p_head;
	int items_freed = 0;
	while (p_current_item) {
		TrainingItem *p_next = p_current_item->p_next; /* Backup pointer to next element */

	    if (p_current_item->fruitName) {	/* Free fruit name member */
	    	//free(p_current_item->fruitName); Only free non-literal strings..
	    }

	    free(p_current_item);				/* Free struct */
	    p_current_item = p_next;			/* Move to the next element */
	    items_freed++;
	}
	printf("Freed %d data runs in total\n", items_freed);
	free(p_head); 					/* Free head */
	return items_freed;
}

/*
 * Reverses the order of the elements in the list with head at *p_head
 * FILO => FIFO, vice versa.
 */
TrainingItem* reverseTList(TrainingItem *p_head) {
  TrainingItem *p_new_head = NULL;
  while (p_head) {
    TrainingItem *p_next = p_head->p_next;
    p_head->p_next = p_new_head;
    p_new_head = p_head;
    p_head = p_next;
  }
  return p_new_head;
}

/**
 * Iterate the list, return the name of the class which has the highest posterior probability.
 * O(n)
 */
char * getMostProbableClass(Posteriors *p_head) {
	double highestP = 0;
	char * class;

	Posteriors *p_current = p_head;
	while(p_current) {
		if(p_current->class && p_current->posteriorP) {
			if(p_current->posteriorP > highestP) {
				highestP = p_current->posteriorP;
				class = p_current->class;
			}
		}
		p_current = p_current->p_next;
	}
	printf("Fruit identified as: %s\n", class);
	return class;
}

#endif /* LINKEDLIST_H_ */
