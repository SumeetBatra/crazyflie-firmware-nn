
#include "network_evaluate.h"
#include "gru_params.h"



float linear(float num) {
	return num;
}



float sigmoid(float num) {
	return 1 / (1 + expf(-num));
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

/*
The gated reccurent unit follows this mechanism:
Reset gate:        r(t) = sigmoid(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
Update gate:       u(t) = sigmoid(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
Cell gate:         c(t) = tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u(t) * c(t)
*/


// =================== Reset gate ====================
// r(t) = sigmoid(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
void compute_reset_gate(const float *state_array) {
	// x(t) @ W_xi
	vec_mul_matrix(intermedia_result_1, state_array, W_xr[0], input_dim, num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, W_hr[0], num_unit, num_unit);
	// add bias and non-linearity
	for (int i = 0; i < num_unit; i++) {
		reset_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + b_r[i]);
	}
	clear_intermedia_results();
}

// ==================== Update gate ======================
// u(t) = sigmoid(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
void compute_update_gate(const float *state_array) {
	// x(t) @ W_xf
	vec_mul_matrix(intermedia_result_1, state_array, W_xu[0], input_dim, num_unit);
	// h(t-1) @ W_hi 
	vec_mul_matrix(intermedia_result_2, hidden_state, W_hu[0], num_unit, num_unit);
	// add bias and non-linearity
	for (int i = 0; i < num_unit; i++) {
		update_gate_t[i] = sigmoid(intermedia_result_1[i] + intermedia_result_2[i] + b_u[i]);
	}
	clear_intermedia_results();
}

// ==================== new cell state ========================
// c(t) = tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
void compute_new_cell_state(const float *state_array) {
	// x(t) @ W_xc
	vec_mul_matrix(intermedia_result_1, state_array, W_xc[0], input_dim, num_unit);
	// h(t-1) @ W_hc
	vec_mul_matrix(intermedia_result_2, hidden_state, W_hc[0], num_unit, num_unit);
	// r(t) * (h(t-1) @ W_hc)
	for (int i = 0; i < num_unit; i++) {
		intermedia_result_2[i] = intermedia_result_2[i] * reset_gate_t[i];
	}
	// tanh(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
	for (int i = 0; i < num_unit; i++) {
		cell_state_t[i] = tanhf(intermedia_result_1[i] + intermedia_result_2[i] + b_c[i]);
	}
	clear_intermedia_results();
}

// ================= new hidden state =====================
// h(t) = (1 - u(t)) * h(t-1) + u(t) * c(t)
void compute_new_hidden_state() {
	// (1 - u(t)) * h(t-1) + u(t) * c(t)
	for (int i = 0; i < num_unit; i++) {
		hidden_state[i] = (1 - update_gate_t[i]) * hidden_state[i] + update_gate_t[i] * cell_state_t[i];
	}
	clear_intermedia_results();
}

void networkEvaluate(struct control_t_n *control_n, const float *state_array) {

	compute_reset_gate(state_array);

	compute_update_gate(state_array);

	compute_new_cell_state(state_array);

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
