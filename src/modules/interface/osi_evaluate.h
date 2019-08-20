#ifndef __OSI_EVALUATE_H__
#define __OSI_EVALUATE_H__

#include <math.h>
#include "network_evaluate.h"

/*
 * osi prediction variables
*/
typedef struct osi_out_n {
	float t2w; 
	float t2t;
} osi_out_n;

void osiEvaluate(osi_out_n *osi_out, const float *state_array);

#endif