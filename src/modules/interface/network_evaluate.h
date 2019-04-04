#ifndef __NETWORK_EVALUATE_H__
#define __NETWORK_EVALUATE_H__

#include <math.h>

/*
 * since the network outputs thrust on each motor,
 * we need to define a struct which stores the values
*/
typedef struct control_t_n {
	float thrust_0; 
	float thrust_1;
	float thrust_2;
	float thrust_3;	
} control_t_n;

void networkEvaluate(control_t_n *control_n, float *state_array);

struct networkWeights
{
	float layer_0_weight[18][64];
	float layer_1_weight[64][64];
	float layer_2_weight[64][4];
	float layer_0_bias[64];
	float layer_1_bias[64];
	float layer_2_bias[4];
} __attribute__((packed));

extern struct networkWeights nn_weights;


#endif