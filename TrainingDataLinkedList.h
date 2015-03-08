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
  float  h; /* Hue */
  float	 s; /* Saturation */
  float	 v; /* Value */
  float  c; /* Compactness */
  struct TrainingItem *p_next;
} TrainingItem;

/**
 * Adds new item to list of training data, returns pointer to new list element
 * as head of list. (i.e. FILO buffer)
 */
TrainingItem* addItem(TrainingItem *p_head, char *fruitName, float h, float s, float v, float c) {
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

/*
 * Traverses the TrainingData and prints off the attributes of each list item.
 * Useful for debugging and testing.
 */
void printList(TrainingItem *p_head) {

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

/*
 * Free all of the TrainingItem elements, and the structure itself.
 */
int freeList(TrainingItem *p_head)	{

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
TrainingItem* reverseList(TrainingItem *p_head) {
  TrainingItem *p_new_head = NULL;
  while (p_head) {
    TrainingItem *p_next = p_head->p_next;
    p_head->p_next = p_new_head;
    p_new_head = p_head;
    p_head = p_next;
  }
  return p_new_head;
}

#endif /* LINKEDLIST_H_ */
