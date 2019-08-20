#include "osi.h"
#include "controller_nn.h"

static float osi_state_array[7];

/*
 * arrange data to what the osi network expects and evaluate the network
 */
void osi_predict(osi_out_n *osi_out, 
				 const float *state_array, 
				 const control_t_n *control_n, 
				 const uint32_t tick) 
{
	if (!RATE_DO_EXECUTE(RATE_100_HZ, tick)) {
		return;
	}

	/* Input to the network:
	 * Vxyz, t0, t1, t2, t3
	 */
	osi_state_array[0] = state_array[3];
	osi_state_array[1] = state_array[4];
	osi_state_array[2] = state_array[5];
	osi_state_array[3] = clip(scale(control_n->thrust_0), 0.0, 1.0);
	osi_state_array[4] = clip(scale(control_n->thrust_1), 0.0, 1.0);
	osi_state_array[5] = clip(scale(control_n->thrust_2), 0.0, 1.0);
	osi_state_array[6] = clip(scale(control_n->thrust_3), 0.0, 1.0);

	osiEvaluate(osi_out, osi_state_array);
}