#ifndef __OSI_H__
#define __OSI_H__

#include "stabilizer_types.h"
#include "network_evaluate.h"
#include "osi_evaluate.h"

/*
 * arrange data to what the osi network expects and evaluate the network
 */
void osi_predict(osi_out_n *osi_out, 
				 const float *state_array, 
				 const control_t_n *control_n, 
				 const uint32_t tick);

#endif