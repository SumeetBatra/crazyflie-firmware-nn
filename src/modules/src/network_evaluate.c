#include "network_evaluate.h"

struct networkDescriptions nn_desc;


/////////////////////////////////////////////////////////////////////
// Helper functions

// static float linear(float num) {
// 	return num;
// }

static float sigmoid(float num) {
	return 1.0f / (1.0f + expf(-num));
}

// static float relu(float num) {
// 	if (num > 0) {
// 		return num;
// 	} else {
// 		return 0;
// 	}
// }

/*
** dim_1: length of the observation
** dim_2: length of the result
*/
static void vec_mul_matrix(float *result, const float *vec, const float *weights, const int dim_1, const int dim_2) {
	for (int i = 0; i < dim_2; i++) {
		for (int j = 0; j < dim_1; j++) {
			// result[i] += vec[j] * weights[j][i];
			result[i] += vec[j] * (*(weights+dim_2*j+i));
		}
	}
}

/////////////////////////////////////////////////////////////////////
// FF Neural Network

// temporary variables
static float output_0[MAX_NN_FF_DIM];
static float output_1[MAX_NN_FF_DIM];
static float output_2[OUTPUT_DIM];

static void ff_networkEvaluate(struct control_t_n *control_n, const float *state_array) {

	for (int i = 0; i < nn_desc.ff.dim; i++) {
		output_0[i] = 0;
		for (int j = 0; j < INPUT_DIM; j++) {
			output_0[i] += state_array[j] * nn_desc.ff.layer_0_weight[j][i];
		}
		output_0[i] += nn_desc.ff.layer_0_bias[i];
		output_0[i] = tanhf(output_0[i]);
	}

	for (int i = 0; i < nn_desc.ff.dim; i++) {
		output_1[i] = 0;
		for (int j = 0; j < nn_desc.ff.dim; j++) {
			output_1[i] += output_0[j] * nn_desc.ff.layer_1_weight[j][i];
		}
		output_1[i] += nn_desc.ff.layer_1_bias[i];
		output_1[i] = tanhf(output_1[i]);
	}
	
	for (int i = 0; i < OUTPUT_DIM; i++) {
		output_2[i] = 0;
		for (int j = 0; j < nn_desc.ff.dim; j++) {
			output_2[i] += output_1[j] * nn_desc.ff.layer_2_weight[j][i];
		}
		output_2[i] += nn_desc.ff.layer_2_bias[i];
	}

	control_n->thrust_0 = output_2[0];
	control_n->thrust_1 = output_2[1];
	control_n->thrust_2 = output_2[2];
	control_n->thrust_3 = output_2[3];
}

/////////////////////////////////////////////////////////////////////
// LSTM Neural Network

/*
Assuming the following machanism:
Incoming gate:    i(t) = sigmoid(x(t) @ W_xi + h(t-1) @ W_hi + b_i)
Forget gate:      f(t) = sigmoid(x(t) @ W_xf + h(t-1) @ W_hf + b_f)
Cell gate:        c(t) = f(t) * c(t - 1) + i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
Out gate:         o(t) = sigmoid(x(t) @ W_xo + h(t-1) @ W_ho + b_o)
New hidden state: h(t) = o(t) * tanh(c(t))
*/

static float hidden_state[MAX_NN_LSTM_DIM];
static float cell_state[MAX_NN_LSTM_DIM];
static float incoming_gate_t[MAX_NN_LSTM_DIM];
static float forget_gate_t[MAX_NN_LSTM_DIM];
static float out_gate_t[MAX_NN_LSTM_DIM];
static float actions[OUTPUT_DIM];
static float intermedia_result_1[MAX_NN_LSTM_DIM];
static float intermedia_result_2[MAX_NN_LSTM_DIM];
static float intermedia_result_3[MAX_NN_LSTM_DIM];
static float intermedia_result_4[MAX_NN_LSTM_DIM];

static void lstm_clear_intermedia_results() {
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		intermedia_result_1[i] = 0;
		intermedia_result_2[i] = 0;
		intermedia_result_3[i] = 0;
		intermedia_result_4[i] = 0;
	}
}

static void lstm_clear_actions() {
	actions[0] = 0;
	actions[1] = 0;
	actions[2] = 0;
	actions[3] = 0;
}

// =================== Incoming gate ====================
// i(t) = sigmoid(x(t) @ W_xi + h(t-1) @ W_hi + b_i)
static void lstm_compute_incoming_gate(const float *state_array) {
	// x(t) @ W_xi
	vec_mul_matrix(intermedia_result_1, state_array, nn_desc.lstm.W_xi[0], INPUT_DIM, nn_desc.lstm.num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, nn_desc.lstm.W_hi[0], nn_desc.lstm.num_unit, nn_desc.lstm.num_unit);
	// add bias and non-linearity
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		incoming_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + nn_desc.lstm.b_i[i]);
	}
	lstm_clear_intermedia_results();
}

// ==================== Forget gate ======================
// f(t) = sigmoid(x(t) @ W_xf + h(t-1) @ W_hf + b_f + forget_bias)
const static float forget_bias = 1.0;
static void lstm_compute_forget_gate(const float *state_array) {
	// x(t) @ W_xf
	vec_mul_matrix(intermedia_result_1, state_array, nn_desc.lstm.W_xf[0], INPUT_DIM, nn_desc.lstm.num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, nn_desc.lstm.W_hf[0], nn_desc.lstm.num_unit, nn_desc.lstm.num_unit);
	// add bias and non-linearity
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		forget_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + nn_desc.lstm.b_f[i] + forget_bias);
	}
	lstm_clear_intermedia_results();
}

// ==================== new cell state ========================
// c(t) = f(t) * c(t - 1) + i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
static void lstm_compute_new_cell_state(const float *state_array) {
	// f(t) * c(t-1), element-wise multiplication
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		intermedia_result_1[i] = forget_gate_t[i] * cell_state[i];
	}
	// x(t) @ W_xc
	vec_mul_matrix(intermedia_result_2, state_array, nn_desc.lstm.W_xc[0], INPUT_DIM, nn_desc.lstm.num_unit);
	// h(t-1) @ W_hc
	vec_mul_matrix(intermedia_result_3, hidden_state, nn_desc.lstm.W_hc[0], nn_desc.lstm.num_unit, nn_desc.lstm.num_unit);
	// tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		intermedia_result_4[i] = tanhf(intermedia_result_2[i] + intermedia_result_3[i] + nn_desc.lstm.b_c[i]);
	}
	// i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		intermedia_result_2[i] = incoming_gate_t[i] * intermedia_result_4[i];
	}
	// f(t) * c(t - 1) + i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		cell_state[i] = intermedia_result_1[i] + intermedia_result_2[i];
	}
	lstm_clear_intermedia_results();
}

// ===================== Out gate =======================
// o(t) = sigmoid(x(t) @ W_xo + h(t-1) @ W_ho + b_o)
static void lstm_compute_out_gate(const float *state_array) {
	// x(t) @ W_xo
	vec_mul_matrix(intermedia_result_1, state_array, nn_desc.lstm.W_xo[0], INPUT_DIM, nn_desc.lstm.num_unit);
	// h(t-1) @ W_ho
	vec_mul_matrix(intermedia_result_2, hidden_state, nn_desc.lstm.W_ho[0], nn_desc.lstm.num_unit, nn_desc.lstm.num_unit);
	// add bias and non-linearity
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		out_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + nn_desc.lstm.b_o[i]);
	}
	lstm_clear_intermedia_results();	
}

// ================= new hidden state =====================
// h(t) = o(t) * tanh(c(t))
static void lstm_compute_new_hidden_state() {
	// tanh(c(t))
	for(int i = 0; i < nn_desc.lstm.num_unit; i++) {
		intermedia_result_1[i] = tanhf(cell_state[i]);
	}
	// o(t) * tanh(c(t))
	for (int i = 0; i < nn_desc.lstm.num_unit; i++) {
		hidden_state[i] = out_gate_t[i] * intermedia_result_1[i];
	}
	lstm_clear_intermedia_results();
}

static void lstm_networkEvaluate(struct control_t_n *control_n, const float *state_array) {

	lstm_compute_incoming_gate(state_array);

	lstm_compute_forget_gate(state_array);

	lstm_compute_new_cell_state(state_array);

	lstm_compute_out_gate(state_array);

	lstm_compute_new_hidden_state();

	// compute actions
	lstm_clear_actions();
	// hidden_state @ W
	vec_mul_matrix(actions, hidden_state, nn_desc.lstm.W[0], nn_desc.lstm.num_unit, 4);
	// add bias
	for (int i = 0; i < 4; i++) {
		actions[i] += nn_desc.lstm.b[i];
	}

	control_n->thrust_0 = actions[0];
	control_n->thrust_1 = actions[1];
	control_n->thrust_2 = actions[2];
	control_n->thrust_3 = actions[3];
}

/////////////////////////////////////////////////////////////////////
// GRU Neural Network

/*
Reset gate:        r(t) = sigmoid(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
Update gate:       u(t) = sigmoid(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
Cell gate:         c(t) = tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u(t) * c(t)
*/

static float hidden_state[MAX_NN_GRU_DIM];
static float reset_gate_t[MAX_NN_GRU_DIM];
static float update_gate_t[MAX_NN_GRU_DIM];
static float cell_state_t[MAX_NN_GRU_DIM];
static float actions[OUTPUT_DIM];
static float intermedia_result_1[MAX_NN_GRU_DIM];
static float intermedia_result_2[MAX_NN_GRU_DIM];

static void gru_clear_intermedia_results() {
	for (int i = 0; i < nn_desc.gru.num_unit; i++) {
		intermedia_result_1[i] = 0;
		intermedia_result_2[i] = 0;
	}
}

static void gru_clear_actions() {
	actions[0] = 0;
	actions[1] = 0;
	actions[2] = 0;
	actions[3] = 0;
}

/*
The gated reccurent unit follows this mechanism:
Reset gate:        r(t) = sigmoid(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
Update gate:       u(t) = sigmoid(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
Cell gate:         c(t) = tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u(t) * c(t)
*/


// =================== Reset gate ====================
// r(t) = sigmoid(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
static void gru_compute_reset_gate(const float *state_array) {
	// x(t) @ W_xi
	vec_mul_matrix(intermedia_result_1, state_array, nn_desc.gru.W_xr[0], INPUT_DIM, nn_desc.gru.num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, nn_desc.gru.W_hr[0], nn_desc.gru.num_unit, nn_desc.gru.num_unit);
	// add bias and non-linearity
	for (int i = 0; i < nn_desc.gru.num_unit; i++) {
		reset_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + nn_desc.gru.b_r[i]);
	}
	gru_clear_intermedia_results();
}

// ==================== Update gate ======================
// u(t) = sigmoid(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
static void gru_compute_update_gate(const float *state_array) {
	// x(t) @ W_xf
	vec_mul_matrix(intermedia_result_1, state_array, nn_desc.gru.W_xu[0], INPUT_DIM, nn_desc.gru.num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, nn_desc.gru.W_hu[0], nn_desc.gru.num_unit, nn_desc.gru.num_unit);
	// add bias and non-linearity
	for (int i = 0; i < nn_desc.gru.num_unit; i++) {
		update_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + nn_desc.gru.b_u[i]);
	}
	gru_clear_intermedia_results();
}

// ==================== new cell state ========================
// c(t) = tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
static void gru_compute_new_cell_state(const float *state_array) {
	// x(t) @ W_xc
	vec_mul_matrix(intermedia_result_1, state_array, nn_desc.gru.W_xc[0], INPUT_DIM, nn_desc.gru.num_unit);
	// h(t-1) @ W_hc
	vec_mul_matrix(intermedia_result_2, hidden_state, nn_desc.gru.W_hc[0], nn_desc.gru.num_unit, nn_desc.gru.num_unit);
	// r(t) * (h(t-1) @ W_hc)
	for (int i = 0; i < nn_desc.gru.num_unit; i++) {
		intermedia_result_2[i] = intermedia_result_2[i] * reset_gate_t[i];
	}
	// tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
	for (int i = 0; i < nn_desc.gru.num_unit; i++) {
		cell_state_t[i] = tanhf(intermedia_result_1[i] + intermedia_result_2[i] + nn_desc.gru.b_c[i]);
	}
	gru_clear_intermedia_results();
}

// ================= new hidden state =====================
// h(t) = (1 - u(t)) * h(t-1) + u(t) * c(t)
static void gru_compute_new_hidden_state() {
	// (1 - u(t)) * h(t-1) + u(t) * c(t)
	for (int i = 0; i < nn_desc.gru.num_unit; i++) {
		hidden_state[i] = (1 - update_gate_t[i]) * hidden_state[i] + update_gate_t[i] * cell_state_t[i];
	}
	gru_clear_intermedia_results();
}

static void gru_networkEvaluate(struct control_t_n *control_n, const float *state_array) {

	gru_compute_reset_gate(state_array);

	gru_compute_update_gate(state_array);

	gru_compute_new_cell_state(state_array);

	gru_compute_new_hidden_state();

	// compute actions
	gru_clear_actions();
	// hidden_state @ W
	vec_mul_matrix(actions, hidden_state, nn_desc.gru.W[0], nn_desc.gru.num_unit, 4);
	// add bias
	for (int i = 0; i < 4; i++) {
		actions[i] += nn_desc.gru.b[i];
	}

	control_n->thrust_0 = actions[0];
	control_n->thrust_1 = actions[1];
	control_n->thrust_2 = actions[2];
	control_n->thrust_3 = actions[3];
}

/////////////////////////////////////////////////////////////////////
// switch function

void networkEvaluate(struct control_t_n *control_n, const float *state_array) {
	switch (nn_desc.networkType) {
		case NetworkTypeFF:
			ff_networkEvaluate(control_n, state_array);
			break;
		case NetworkTypeLSTM:
			lstm_networkEvaluate(control_n, state_array);
			break;
		case NetworkTypeGRU:
			gru_networkEvaluate(control_n, state_array);
			break;
		default:
			break;
	}
}