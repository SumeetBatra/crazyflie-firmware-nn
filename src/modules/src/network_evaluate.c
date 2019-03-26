
#include "network_evaluate.h"
#include "lstm_params.h"



float linear(float num) {
	return num;
}



float sigmoid(float num) {
	return 1.0f / (1.0f + expf(-num));
}



float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

void clear_intermedia_results() {
	for (int i = 0; i < num_unit; i++) {
		intermedia_result_1[i] = 0;
		intermedia_result_2[i] = 0;
		intermedia_result_3[i] = 0;
		intermedia_result_4[i] = 0;
	}
}

void clear_actions() {
	actions[0] = 0;
	actions[1] = 0;
	actions[2] = 0;
	actions[3] = 0;
}

/*
** dim_1: length of the observation
** dim_2: length of the result
*/
void vec_mul_matrix(float *result, const float *vec, const float *weights, const int dim_1, const int dim_2) {
	for (int i = 0; i < dim_2; i++) {
		for (int j = 0; j < dim_1; j++) {
			// result[i] += vec[j] * weights[j][i];
			result[i] += vec[j] * (*(weights+dim_2*j+i));
		}
	}
}

// =================== Incoming gate ====================
// i(t) = sigmoid(x(t) @ W_xi + h(t-1) @ W_hi + b_i)
void compute_incoming_gate(const float *state_array) {
	// x(t) @ W_xi
	vec_mul_matrix(intermedia_result_1, state_array, W_xi[0], input_dim, num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, W_hi[0], num_unit, num_unit);
	// add bias and non-linearity
	for (int i = 0; i < num_unit; i++) {
		incoming_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + b_i[i]);
	}
	clear_intermedia_results();
}

// ==================== Forget gate ======================
// f(t) = sigmoid(x(t) @ W_xf + h(t-1) @ W_hf + b_f + forget_bias)
const static float forget_bias = 1.0;
void compute_forget_gate(const float *state_array) {
	// x(t) @ W_xf
	vec_mul_matrix(intermedia_result_1, state_array, W_xf[0], input_dim, num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, W_hf[0], num_unit, num_unit);
	// add bias and non-linearity
	for (int i = 0; i < num_unit; i++) {
		forget_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + b_f[i] + forget_bias);
	}
	clear_intermedia_results();
}

// ==================== new cell state ========================
// c(t) = f(t) * c(t - 1) + i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
void compute_new_cell_state(const float *state_array) {
	// f(t) * c(t-1), element-wise multiplication
	for (int i = 0; i < num_unit; i++) {
		intermedia_result_1[i] = forget_gate_t[i] * cell_state[i];
	}
	// x(t) @ W_xc
	vec_mul_matrix(intermedia_result_2, state_array, W_xc[0], input_dim, num_unit);
	// h(t-1) @ W_hc
	vec_mul_matrix(intermedia_result_3, hidden_state, W_hc[0], num_unit, num_unit);
	// tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
	for (int i = 0; i < num_unit; i++) {
		intermedia_result_4[i] = tanhf(intermedia_result_2[i] + intermedia_result_3[i] + b_c[i]);
	}
	// i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
	for (int i = 0; i < num_unit; i++) {
		intermedia_result_2[i] = incoming_gate_t[i] * intermedia_result_4[i];
	}
	// f(t) * c(t - 1) + i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
	for (int i = 0; i < num_unit; i++) {
		cell_state[i] = intermedia_result_1[i] + intermedia_result_2[i];
	}
	clear_intermedia_results();
}

// ===================== Out gate =======================
// o(t) = sigmoid(x(t) @ W_xo + h(t-1) @ W_ho + b_o)
void compute_out_gate(const float *state_array) {
	// x(t) @ W_xo
	vec_mul_matrix(intermedia_result_1, state_array, W_xo[0], input_dim, num_unit);
	// h(t-1) @ W_ho
	vec_mul_matrix(intermedia_result_2, hidden_state, W_ho[0], num_unit, num_unit);
	// add bias and non-linearity
	for (int i = 0; i < num_unit; i++) {
		out_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + b_o[i]);
	}
	clear_intermedia_results();	
}

// ================= new hidden state =====================
// h(t) = o(t) * tanh(c(t))
void compute_new_hidden_state() {
	// tanh(c(t))
	for(int i = 0; i < num_unit; i++) {
		intermedia_result_1[i] = tanhf(cell_state[i]);
	}
	// o(t) * tanh(c(t))
	for (int i = 0; i < num_unit; i++) {
		hidden_state[i] = out_gate_t[i] * intermedia_result_1[i];
	}
	clear_intermedia_results();
}

void networkEvaluate(struct control_t_n *control_n, const float *state_array) {

	compute_incoming_gate(state_array);

	compute_forget_gate(state_array);

	compute_new_cell_state(state_array);

	compute_out_gate(state_array);

	compute_new_hidden_state();

	// compute actions
	clear_actions();
	// hidden_state @ W
	vec_mul_matrix(actions, hidden_state, W[0], num_unit, 4);
	// add bias
	for (int i = 0; i < 4; i++) {
		actions[i] += b[i];
	}

	control_n->thrust_0 = actions[0];
	control_n->thrust_1 = actions[1];
	control_n->thrust_2 = actions[2];
	control_n->thrust_3 = actions[3];	

}
