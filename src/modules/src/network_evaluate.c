
#include "network_evaluate.h"



float linear(float num) {
	return num;
}



float sigmoid(float num) {
	return 1 / (1 + exp(-num));
}



float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

static const int structure[3][2] = {{18, 64},{64, 64},{64, 4}};
static float output_0[64];
static float output_1[64];
static float output_2[4];

struct networkWeights nn_weights;

	void networkEvaluate(struct control_t_n *control_n, float *state_array) {
	
		for (int i = 0; i < structure[0][1]; i++) {
			output_0[i] = 0;
			for (int j = 0; j < structure[0][0]; j++) {
				output_0[i] += state_array[j] * nn_weights.layer_0_weight[j][i];
			}
			output_0[i] += nn_weights.layer_0_bias[i];
			output_0[i] = tanhf(output_0[i]);
		}
	
		for (int i = 0; i < structure[1][1]; i++) {
			output_1[i] = 0;
			for (int j = 0; j < structure[1][0]; j++) {
				output_1[i] += output_0[j] * nn_weights.layer_1_weight[j][i];
			}
			output_1[i] += nn_weights.layer_1_bias[i];
			output_1[i] = tanhf(output_1[i]);
		}
		
		for (int i = 0; i < structure[2][1]; i++) {
			output_2[i] = 0;
			for (int j = 0; j < structure[2][0]; j++) {
				output_2[i] += output_1[j] * nn_weights.layer_2_weight[j][i];
			}
			output_2[i] += nn_weights.layer_2_bias[i];
		}
		
		control_n->thrust_0 = output_2[0];
		control_n->thrust_1 = output_2[1];
		control_n->thrust_2 = output_2[2];
		control_n->thrust_3 = output_2[3];	
	
	}
	