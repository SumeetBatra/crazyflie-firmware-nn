#ifndef __NETWORK_EVALUATE_H__
#define __NETWORK_EVALUATE_H__

#include <math.h>
#include <stdint.h>

#define INPUT_DIM (18)
#define OUTPUT_DIM (4)

#define MAX_NN_FF_DIM (64)

#define MAX_NN_LSTM_DIM (32)
#define MAX_NN_GRU_DIM (32)

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

void networkEvaluate(control_t_n *control_n, const float *state_array);

struct networkWeightsFF
{
	uint16_t dim;
	float layer_0_weight[INPUT_DIM][MAX_NN_FF_DIM];
	float layer_1_weight[MAX_NN_FF_DIM][MAX_NN_FF_DIM];
	float layer_2_weight[MAX_NN_FF_DIM][OUTPUT_DIM];
	float layer_0_bias[MAX_NN_FF_DIM];
	float layer_1_bias[MAX_NN_FF_DIM];
	float layer_2_bias[OUTPUT_DIM];
} __attribute__((packed));

struct networkWeightsLSTM
{
	uint16_t num_unit;
	float W_xi[INPUT_DIM][MAX_NN_LSTM_DIM];
	float W_hi[MAX_NN_LSTM_DIM][MAX_NN_LSTM_DIM];
	float b_i[MAX_NN_LSTM_DIM];
	float W_xf[INPUT_DIM][MAX_NN_LSTM_DIM];
	float W_hf[MAX_NN_LSTM_DIM][MAX_NN_LSTM_DIM];
	float b_f[MAX_NN_LSTM_DIM];
	float W_xc[INPUT_DIM][MAX_NN_LSTM_DIM];
	float W_hc[MAX_NN_LSTM_DIM][MAX_NN_LSTM_DIM];
	float b_c[MAX_NN_LSTM_DIM];
	float W_xo[INPUT_DIM][MAX_NN_LSTM_DIM];
	float W_ho[MAX_NN_LSTM_DIM][MAX_NN_LSTM_DIM];
	float b_o[MAX_NN_LSTM_DIM];
	float W[MAX_NN_LSTM_DIM][OUTPUT_DIM];
	float b[OUTPUT_DIM];
} __attribute__((packed));

struct networkWeightsGRU
{
	uint16_t num_unit;
	float W_xr[INPUT_DIM][MAX_NN_GRU_DIM];
	float W_hr[MAX_NN_GRU_DIM][MAX_NN_GRU_DIM];
	float b_r[MAX_NN_GRU_DIM];
	float W_xu[INPUT_DIM][MAX_NN_GRU_DIM];
	float W_hu[MAX_NN_GRU_DIM][MAX_NN_GRU_DIM];
	float b_u[MAX_NN_GRU_DIM];
	float W_xc[INPUT_DIM][MAX_NN_GRU_DIM];
	float W_hc[MAX_NN_GRU_DIM][MAX_NN_GRU_DIM];
	float b_c[MAX_NN_GRU_DIM];
	float W[MAX_NN_GRU_DIM][OUTPUT_DIM];
	float b[OUTPUT_DIM];
} __attribute__((packed));

enum networkType
{
	NetworkTypeFF     = 0,
	NetworkTypeLSTM   = 1,
	NetworkTypeGRU    = 2,
};

struct networkDescriptions
{
	uint8_t networkType; // one of enum networkType
	union
	{
		struct networkWeightsFF ff;
		struct networkWeightsLSTM lstm;
		struct networkWeightsGRU gru;
	};
} __attribute__((packed));

extern struct networkDescriptions nn_desc;


#endif