import math

import numpy as np
import tensorflow as tf
from sionna.mimo import lmmse_equalizer


class RX_ops_and_losses():

    def __init__(self, constellation, demapper, setup):
        super(RX_ops_and_losses, self).__init__()
        self.constellation = constellation
        self.demapper = demapper
        self.setup = setup
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def cyclical_shift(self, inputs):
        Lambda_matrix, k, flip = inputs
        return tf.cond(tf.equal(flip, True),
                       lambda: tf.roll(tf.reverse(Lambda_matrix, axis=[0]), shift=tf.squeeze(k) + 1, axis=0),
                       lambda: tf.roll(Lambda_matrix, shift=tf.squeeze(k), axis=0))

    @tf.function
    def non_zero_element_finder_for_H_tilde(self, k, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(
            self.setup.K_prime / 2. - z * self.setup.K_prime / 2.)  # original position of zero starting in the fft sequence of phase noise
        # print('tf.range(int(self.setup.K * z)): ', tf.range(int(self.setup.K * z)))
        ZI = tf.math.floormod(B_orig + tf.range(int(self.setup.K_prime * z)),
                              self.setup.K_prime)  # zero indices for k-rolled fft sequence of phase noise
        # ZI = tf.math.floormod(B_orig + np.array(range(int(self.setup.K * z))), self.setup.K)  # zero indices for k-rolled fft sequence of phase noise
        ZI = tf.cast(ZI, dtype=tf.int64)
        s = ZI.shape
        mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
                                                                               values=tf.ones(shape=[s[0]],
                                                                                              dtype=tf.int32),
                                                                               dense_shape=[self.setup.K_prime]))
        mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)
        mask_of_ones_after_shift_flip_true = tf.roll(tf.reverse(mask_of_ones_before_shift, axis=[0]),
                                                     shift=tf.squeeze(k) + 1, axis=0)
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(k), axis=0)
        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total

    @tf.function
    def H_tilde_k_q(self, bundeled_inputs_0):
        H_masked, Lambda_B_cyclshifted_masked, Lambda_U_cyclshifted_masked = bundeled_inputs_0
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted_masked,
                                                 H_masked),
                                Lambda_B_cyclshifted_masked)

    @tf.function
    def non_zero_element_finder_for_H_hat(self, k, m, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(
            self.setup.K_prime / 2. - z * self.setup.K_prime / 2.)  # original position of zero starting in the fft sequence of phase noise
        ZI = tf.math.floormod(B_orig + tf.range(int(self.setup.K_prime * z)),
                              self.setup.K_prime)  # zero indices for k-rolled fft sequence of phase noise
        # ZI = tf.math.floormod(B_orig + np.array(range(int(self.setup.K * z))), self.setup.K)  # zero indices for k-rolled fft sequence of phase noise
        ZI = tf.cast(ZI, dtype=tf.int64)
        s = ZI.shape
        mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
                                                                               values=tf.ones(shape=[s[0]],
                                                                                              dtype=tf.int32),
                                                                               dense_shape=[self.setup.K_prime]))
        mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)

        mask_of_ones_after_shift_flip_true = tf.roll(tf.reverse(mask_of_ones_before_shift, axis=[0]),
                                                     shift=tf.squeeze(k) + 1, axis=0)
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(m), axis=0)

        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total

    @tf.function
    def matrix_multplication(self, inputs):
        vd_forall_u, s_forall_u = inputs
        return tf.linalg.matmul(vd_forall_u, s_forall_u)

    @tf.function
    def mu_superposition_beamformer(self, inputs):
        vd_forall_u, s_forall_u = inputs
        return tf.reduce_sum(tf.map_fn(self.matrix_multplication, [vd_forall_u, s_forall_u],
                                       fn_output_signature=tf.complex64,
                                       parallel_iterations=self.setup.Nue),
                             axis=0)

    @tf.function
    def received_sig_m_k(self, bundeled_inputs_0):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # S [Nue, N_s, 1]
        # V_D [Nue, N_b_rf, N_s]
        # W_D [N_u_rf, N_s]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # H_hat [N_u_a, N_u_b]

        S_m_forall_u, V_D_m_forall_u, W_D_k, V_RF, W_RF, H_hat, k, m = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        T1 = tf.linalg.matmul(T0, H_hat)
        T2 = tf.linalg.matmul(T1, V_RF)
        S_m_forall_u = tf.reshape(S_m_forall_u, shape=[self.setup.Nue, self.setup.N_s, 1])
        T3 = self.mu_superposition_beamformer([V_D_m_forall_u, S_m_forall_u])
        rx_sig = tf.linalg.matmul(T2, T3)
        return rx_sig

    @tf.function
    def H_hat_k_m_q(self, inputs):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # H [K_prime, N_u_a, N_u_b]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]
        # k [K_prime]
        # m [K_prime]

        H, Lambda_B_cyclshifted, Lambda_U_cyclshifted, k, m = inputs
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted, H), Lambda_B_cyclshifted)

    @tf.function
    def H_hat_k_m(self, inputs):
        # HOW THE INPUTS COME IN -----------------------------------------------
        # H [K_prime, N_u_a, N_u_b]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]
        # k 1
        # m 1

        H, Lambda_B, Lambda_U, k, m = inputs
        Lambda_B_cyclshifted = self.cyclical_shift([Lambda_B, tf.squeeze(m), False])
        Lambda_U_cyclshifted = self.cyclical_shift([Lambda_U, tf.squeeze(k), True])
        k_repeated = tf.tile([k], multiples=[self.setup.K_prime])
        m_repeated = tf.tile([m], multiples=[self.setup.K_prime])

        # HOW THE INPUTS ARE CONSUMED -------------------------------------------
        # H [K_prime, N_u_a, N_u_b]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]
        # k [K_prime]
        # m [K_prime]

        inputs1 = [H, Lambda_B_cyclshifted, Lambda_U_cyclshifted, k_repeated, m_repeated]
        return tf.reduce_sum(tf.map_fn(self.H_hat_k_m_q,
                                       inputs1,
                                       fn_output_signature=tf.complex64,
                                       parallel_iterations=self.setup.K_prime), axis=0)  # sums over q

    @tf.function
    def N_Q_m_k(self, bundeled_inputs_0):
        Lambda_U_k_sub_m_mod_K, W_D_k, W_RF = bundeled_inputs_0
        Z_k = tf.complex(tf.random.normal(shape=[self.setup.N_u_a, 1], mean=0.0, stddev=np.sqrt(self.setup.sigma2 / 2)),
                         tf.random.normal(shape=[self.setup.N_u_a, 1], mean=0.0, stddev=np.sqrt(self.setup.sigma2 / 2)))

        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        T1 = tf.linalg.matmul(T0, Lambda_U_k_sub_m_mod_K, adjoint_a=False, adjoint_b=False)
        y_awgn = tf.linalg.matmul(T1, Z_k, adjoint_a=False, adjoint_b=False)
        return y_awgn

    @tf.function
    def Y_per_k(self, bundeled_inputs_0):
        S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0  # [k, ...]
        # HOW THE INPUTS COME IN -----------------------------------------------
        # S [K_prime, Nue, N_s, log2(M)]
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H [K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]

        V_D_transposed = tf.transpose(V_D, perm=[1, 0, 2, 3])
        W_D_k_repeated = tf.tile([W_D[k, :, :]], multiples=[self.setup.K_prime, 1, 1])
        H_repeated = tf.tile([H], multiples=[self.setup.K_prime, 1, 1, 1])
        V_RF_repeated = tf.tile([V_RF], multiples=[self.setup.K_prime, 1, 1])
        W_RF_repeated = tf.tile([W_RF], multiples=[self.setup.K_prime, 1, 1])
        Lambda_B_repeated = tf.tile([Lambda_B], multiples=[self.setup.K_prime, 1, 1, 1])
        Lambda_U_repeated = tf.tile([Lambda_U], multiples=[self.setup.K_prime, 1, 1, 1])
        k_repeated = tf.tile([k], multiples=[self.setup.K_prime])
        all_m = tf.range(self.setup.K_prime)

        # HOW THE INPUTS ARE CONSUMED -------------------------------------------
        # S [K_prime, Nue, N_s, 1]
        # V_D [K_prime, Nue, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H [K_prime, K_prime, N_u_a, N_u_b]
        # V_RF [K_prime, N_b_a, N_b_rf]
        # W_RF [K_prime, N_u_a, N_u_rf]
        # Lambda_B  [K_prime, K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, K_prime, N_u_a, N_u_a]
        # H_hat_forall_m [K_prime, N_u_a, N_u_b]

        # NOTE, THE ALL MATRICES ARE REPEATED SO all_m CAN BE USED AS A VARIABLE

        bundeled_inputs_1 = [H_repeated, Lambda_B_repeated, Lambda_U_repeated, k_repeated, all_m]
        H_hat_forall_m = tf.map_fn(self.H_hat_k_m, bundeled_inputs_1, fn_output_signature=tf.complex64)  # forall m

        bundeled_inputs_2 = [S, V_D_transposed, W_D_k_repeated, V_RF_repeated, W_RF_repeated, H_hat_forall_m,
                             k_repeated, all_m]

        y_k = tf.reduce_sum(tf.map_fn(self.received_sig_m_k,
                                      bundeled_inputs_2,
                                      fn_output_signature=tf.complex64,
                                      parallel_iterations=self.setup.K_prime), axis=0)

        bundeled_inputs_2 = [self.cyclical_shift([Lambda_U, tf.squeeze(k), True]), W_D_k_repeated, W_RF_repeated]
        awgn_k = tf.reduce_sum(tf.map_fn(self.N_Q_m_k,
                                         bundeled_inputs_2,
                                         fn_output_signature=tf.complex64,
                                         parallel_iterations=round(self.setup.K_prime * self.setup.truncation_ratio_keep)),
                               axis=0)

        return tf.add(y_k, awgn_k)

    @tf.function
    def Y_for_k_equal_to_0(self, bundeled_inputs_0):
        S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [k, ...]
        # HOW THE INPUTS COME IN -----------------------------------------------
        # S [K_prime, Nue, N_s, 1]
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H [K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]

        # HOW THE INPUTS ARE CONSUMED --------------------------------------------
        # S [K_prime, Nue, N_s, log2(M)]
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H [K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]

        # S = tf.tile([S], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1])
        # V_D = tf.tile([V_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1, 1])
        # W_D = tf.tile([W_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1, 1])
        # H = tf.tile([H], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1, 1])
        # V_RF = tf.tile([V_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1])
        # W_RF = tf.tile([W_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1])
        # Lambda_B = tf.tile([Lambda_B],
        #                    multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1, 1])
        # Lambda_U = tf.tile([Lambda_U],
        #                    multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K_prime), 1, 1, 1])
        bundeled_inputs_1 = [S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, 0]
        y = self.Y_per_k(bundeled_inputs_1)
        return y

    @tf.function
    def Y_for_all_M(self, bundeled_inputs_0):
        S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0
        # HOW THE INPUTS COME IN -----------------------------------------------
        # S [M, K_prime, Nue, N_s, 1]
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H [K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [K_prime, N_u_a, N_u_a]

        V_D = tf.tile([V_D], multiples=[self.setup.M, 1, 1, 1, 1])
        W_D = tf.tile([W_D], multiples=[self.setup.M, 1, 1, 1])
        H = tf.tile([H], multiples=[self.setup.M, 1, 1, 1])
        V_RF = tf.tile([V_RF], multiples=[self.setup.M, 1, 1])
        W_RF = tf.tile([W_RF], multiples=[self.setup.M, 1, 1])
        Lambda_B = tf.tile([Lambda_B],
                           multiples=[self.setup.M, 1, 1, 1])
        Lambda_U = tf.tile([Lambda_U],
                           multiples=[self.setup.M, 1, 1, 1])
        bundeled_inputs_1 = [S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]

        # HOW THE INPUTS ARE CONSUMED ------------------------------------------
        # S [M, K_prime, Nue, N_s, 1]
        # V_D [M, Nue, K_prime, N_b_rf, N_s]
        # W_D [M, K_prime, N_u_rf, N_s]
        # H [M, K_prime, N_u_a, N_u_b]
        # V_RF [M, N_b_a, N_b_rf]
        # W_RF [M, N_u_a, N_u_rf]
        # Lambda_B  [M, K_prime, N_b_a, N_b_a]
        # Lambda_U [M, K_prime, N_u_a, N_u_a]

        y = tf.map_fn(self.Y_for_k_equal_to_0, bundeled_inputs_1, fn_output_signature=tf.complex64,
                      parallel_iterations=self.setup.M)
        return y

    @tf.function
    def Y_forall_ues(self, inputs_0):
        S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = inputs_0
        # HOW THE INPUTS COME IN --------------------------------------------------
        # S [M, K_prime, Nue, N_s, 1]
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [Nue, K_prime, N_u_rf, N_s]
        # H [Nue, K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [Nue, N_u_a, N_u_rf]
        # Lambda_B  [K_prime, N_b_a, N_b_a]
        # Lambda_U [Nue, K_prime, N_u_a, N_u_a]

        S = tf.tile([S], multiples=[self.setup.Nue, 1, 1, 1, 1, 1])
        V_D = tf.tile([V_D], multiples=[self.setup.Nue, 1, 1, 1, 1])
        V_RF = tf.tile([V_RF], multiples=[self.setup.Nue, 1, 1])
        Lambda_B = tf.tile([Lambda_B], multiples=[self.setup.Nue, 1, 1, 1])

        inputs_1 = [S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
        # HOW THE INPUTS ARE CONSUMED-----------------------------------------------
        # S [Nue, M, K_prime, Nue, N_s, 1]
        # V_D [Nue, Nue, K_prime, N_b_rf, N_s]
        # W_D [Nue, K_prime, N_u_rf, N_s]
        # H [Nue, K_prime, N_u_a, N_u_b]
        # V_RF [Nue, N_b_a, N_b_rf]
        # W_RF [Nue, N_u_a, N_u_rf]
        # Lambda_B  [Nue, K_prime, N_b_a, N_b_a]
        # Lambda_U [Nue, K_prime, N_u_a, N_u_a]

        y = tf.map_fn(self.Y_for_all_M, inputs_1,
                      fn_output_signature=tf.complex64,
                      parallel_iterations=self.setup.Nue)
        return y

    @tf.function
    def Y_forall_symbols(self, bundeled_inputs_0):
        S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0
        # HOW THE INPUTS COME IN --------------------------------------------------
        # S [int(Nsymb * sampling_ratio_time_domain_keep), M, K_prime, Nue, N_s, log2(M)]
        # V_D [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_b_rf, N_s]
        # W_D [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_rf, N_s]
        # H [Nue, K, N_u_a, N_u_b]
        # V_RF [int(Nsymb * sampling_ratio_time_domain_keep), N_b_a, N_b_rf]
        # W_RF [int(Nsymb * sampling_ratio_time_domain_keep), Nue, N_u_a, N_u_rf]
        # Lambda_B  [int(Nsymb * sampling_ratio_time_domain_keep), K_prime, N_b_a, N_b_a]
        # Lambda_U [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_a, N_u_a]

        H = tf.tile([H], multiples=[round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep), 1, 1, 1, 1])
        bundeled_inputs_1 = [S, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]

        # HOW THE INPUTS ARE CONSUMED ----------------------------------------------
        # V_D [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_b_rf, N_s]
        # W_D [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_rf, N_s]
        # H [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K, N_u_a, N_u_b]
        # V_RF [int(Nsymb * sampling_ratio_time_domain_keep), N_b_a, N_b_rf]
        # W_RF [int(Nsymb * sampling_ratio_time_domain_keep), Nue, N_u_a, N_u_rf]
        # Lambda_B  [int(Nsymb * sampling_ratio_time_domain_keep), K_prime, N_b_a, N_b_a]
        # Lambda_U [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_a, N_u_a]

        y = tf.map_fn(self.Y_forall_ues, bundeled_inputs_1,
                      fn_output_signature=tf.complex64,
                      parallel_iterations=round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep))
        return y

    @tf.function
    def Y_forall_samples(self, bundeled_inputs_0):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # tx_symbols [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), M, K_prime, Nue, N_s, log2(M)]
        # V_D [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_b_rf, N_s]
        # W_D [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_rf, N_s]
        # H [BATCHSIZE, Nue, K_prime, N_u_a, N_u_b]
        # V_RF [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), N_b_a, N_b_rf]
        # W_RF [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, N_u_a, N_u_rf]
        # Lambda_B  [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), K_prime, N_b_a, N_b_a]
        # Lambda_U [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_a, N_u_a]

        y = tf.map_fn(self.Y_forall_symbols, bundeled_inputs_0,
                      fn_output_signature=tf.complex64,
                      parallel_iterations=self.setup.BATCHSIZE)
        return y

    @tf.function
    def Frobinious_distance_of_modulated_signals(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, tx_bits, tx_symbols = bundeled_inputs_0
        bundeled_inputs_1 = [tx_symbols, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
        y_symbols_ = self.Y_forall_samples(bundeled_inputs_1)
        diff = tf.reshape(tf.squeeze(y_symbols_), [-1, 1]) - tf.reshape(tf.squeeze(tx_symbols[:, :, 0, :]), [-1, 1])
        l2 = tf.norm(diff, ord='euclidean', axis=None)
        return l2, y_symbols_

    @tf.function
    def rx_calc(self, bundeled_inputs_0):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # V_D [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, Kprime, N_b_rf, N_s]
        # W_D [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, Kprime, N_u_rf, N_s]
        # H [BATCHSIZE, Nue, K_prime, N_u_a, N_u_b]
        # V_RF [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), N_b_a, N_b_rf]
        # W_RF [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, N_u_a, N_u_rf]
        # Lambda_B  [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), K_prime, N_b_a, N_b_a]
        # Lambda_U [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_a, N_u_a]
        # tx_bits [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), M, K_prime, Nue, N_s, log2(M)]
        # tx_symbols [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), M, K_prime, Nue, N_s, 1]
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, tx_bits, tx_symbols = bundeled_inputs_0
        bundeled_inputs_1 = [tx_symbols, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
        y_symbols = self.Y_forall_samples(bundeled_inputs_1)
        return y_symbols

    @tf.function
    def equalizer_per_k(self, inputs):
        V_D, W_D, H_tilde, V_RF, W_RF, y = inputs
        # HOW THE INPUTS COME IN --------------------------------------------------
        # V_D [N_b_rf, N_s]
        # W_D [N_u_rf, N_s]
        # H_tilde [N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # y [N_s, 1]

        T0 = tf.linalg.matmul(W_D, W_RF, adjoint_a=True, adjoint_b=True)
        T1 = tf.linalg.matmul(T0, H_tilde)
        T2 = tf.linalg.matmul(T1, V_RF)
        T3 = tf.linalg.matmul(T2, V_D)
        y_eq, n = lmmse_equalizer(tf.transpose(y, perm=[1, 0]), T3,
                                  self.setup.sigma2 / (2 * np.pi) * tf.eye(self.setup.N_s, dtype=tf.complex64),
                                  whiten_interference=False)

        return tf.transpose(y_eq, perm=[1, 0]), n

    @tf.function
    def equalizer_for_k_0(self, inputs):
        V_D, W_D, H_tilde, V_RF, W_RF, y = inputs
        # HOW THE INPUTS COME IN --------------------------------------------------
        # V_D [K_prime, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H_tilde [K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # y [N_s, 1]

        # FOR K=0
        # note that y is only calculated for K=0 already, so it should not be sliced here
        inputs1 = [V_D[0, :, :], W_D[0, :, :], H_tilde[0, :, :], V_RF, W_RF, y]
        return self.equalizer_per_k(inputs1)

    @tf.function
    def equalizer_forall_M(self, inputs):
        V_D, W_D, H_tilde, V_RF, W_RF, y = inputs
        # HOW THE INPUTS COME IN -----------------------------------------------
        # V_D [K_prime, N_b_rf, N_s]
        # W_D [K_prime, N_u_rf, N_s]
        # H_tilde [K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [N_u_a, N_u_rf]
        # y [M, N_s, 1]

        V_D = tf.tile([V_D], multiples=[self.setup.M, 1, 1, 1])
        W_D = tf.tile([W_D], multiples=[self.setup.M, 1, 1, 1])
        H_tilde = tf.tile([H_tilde], multiples=[self.setup.M, 1, 1, 1])
        V_RF = tf.tile([V_RF], multiples=[self.setup.M, 1, 1])
        W_RF = tf.tile([W_RF], multiples=[self.setup.M, 1, 1])

        bundeled_inputs_1 = [V_D, W_D, H_tilde, V_RF, W_RF, y]

        # HOW THE INPUTS ARE CONSUMED ------------------------------------------
        # V_D [M, K_prime, N_b_rf, N_s]
        # W_D [M, K_prime, N_u_rf, N_s]
        # H_tilde [M, K_prime, N_u_a, N_u_b]
        # V_RF [M, N_b_a, N_b_rf]
        # W_RF [M, N_u_a, N_u_rf]
        # y [M, N_s, 1]

        return tf.map_fn(self.equalizer_for_k_0,
                         bundeled_inputs_1,
                         fn_output_signature=(tf.complex64, tf.float32),
                         parallel_iterations=self.setup.K_prime)

    @tf.function
    def equalizer_forall_u(self, inputs):
        V_D, W_D, H_tilde, V_RF, W_RF, y = inputs
        # HOW THE INPUTS COME IN --------------------------------------------------
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [Nue, K_prime, N_u_rf, N_s]
        # H_tilde [Nue, K_prime, N_u_a, N_u_b]
        # V_RF [N_b_a, N_b_rf]
        # W_RF [Nue, N_u_a, N_u_rf]
        # y [Nue, M, N_s, 1]

        V_RF = tf.tile([V_RF], multiples=[self.setup.Nue, 1, 1])

        inputs_1 = [V_D, W_D, H_tilde, V_RF, W_RF, y]

        # HOW THE INPUTS ARE CONSUMED --------------------------------------------
        # V_D [Nue, K_prime, N_b_rf, N_s]
        # W_D [Nue, K_prime, N_u_rf, N_s]
        # H_tilde [Nue, K_prime, N_u_a, N_u_b]
        # V_RF [Nue, N_b_a, N_b_rf]
        # W_RF [Nue, N_u_a, N_u_rf]
        # y [Nue, M, N_s, 1]

        return tf.map_fn(self.equalizer_forall_M,
                         inputs_1,
                         fn_output_signature=(tf.complex64, tf.float32),
                         parallel_iterations=self.setup.Nue)

    @tf.function
    def equalizer_forall_symbols(self, inputs):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # V_D [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_b_rf, N_s]
        # W_D [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_rf, N_s]
        # H_tilde [int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_a, N_u_b]
        # V_RF [int(Nsymb * sampling_ratio_time_domain_keep), N_b_a, N_b_rf]
        # W_RF [int(Nsymb * sampling_ratio_time_domain_keep), Nue, N_u_a, N_u_rf]
        # y [int(Nsymb * sampling_ratio_time_domain_keep), Nue, M, K_prime, N_s, 1]

        return tf.map_fn(self.equalizer_forall_u,
                         inputs,
                         fn_output_signature=(tf.complex64, tf.float32),
                         parallel_iterations=round(self.setup.sampling_ratio_time_domain_keep * self.setup.CSIRSPeriod))

    @tf.function
    def equalizer_forall_samples(self, inputs):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # V_D [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_b_rf, N_s]
        # W_D [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_rf, N_s]
        # H_tilde [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, K_prime, N_u_a, N_u_b]
        # V_RF [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), N_b_a, N_b_rf]
        # W_RF [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, N_u_a, N_u_rf]
        # y [BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, M, K_prime, N_s, 1]

        return tf.map_fn(self.equalizer_forall_symbols,
                         inputs,
                         fn_output_signature=(tf.complex64, tf.float32),
                         parallel_iterations=self.setup.BATCHSIZE)

    @tf.function
    def BCE_loss(self, bundeled_inputs_0):
        b, llr = bundeled_inputs_0
        # b:   BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, M 2, Nue 3, N_s 4, log2(M) 5
        # llr: BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, Nue 3, M 2, N_s 4, log2(M) 5
        b = tf.transpose(b, [0, 1, 3, 2, 4, 5])
        # b: BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, Nue 3, M 2, N_s 4, log2(M) 5
        return self.bce(b, llr)

    @tf.function
    def BMI_for_k_0(self, inputs):
        b, llr = inputs
        llr = tf.clip_by_value(llr, -30., 30.)

        # if np.count_nonzero(np.isnan(1 - self.bce(b, llr) / tf.math.log(2.))) != 0:

        #     print(llr)
        #     print(b)

        return 1 - self.bce(b, llr) / tf.math.log(2.)

    # np.count_nonzero(np.isnan(b))
    @tf.function
    def BMI_forall_ues(self, inputs):
        return tf.map_fn(self.BMI_for_k_0, inputs, fn_output_signature=tf.float32)

    @tf.function
    def BMI_forall_symbols(self, inputs):
        return tf.map_fn(self.BMI_forall_ues, inputs, fn_output_signature=tf.float32)

    @tf.function
    def BMI_forall_samples(self, inputs):
        return tf.map_fn(self.BMI_forall_symbols, inputs, fn_output_signature=tf.float32)

    @tf.function
    def BMI_metric(self, inputs):
        # b:   BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), M, Nue, N_s, log2(M)
        # llr: BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, M, N_s, log2(M)
        b, llr = inputs
        b = tf.transpose(b, [0, 1, 3, 2, 4, 5])
        bmi = self.BMI_forall_samples([b, llr])
        return tf.reduce_mean(bmi), bmi

    @tf.function
    def BMI_loss(self, inputs):
        # b:   BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), M, Nue, N_s, log2(M)
        # llr: BATCHSIZE, int(Nsymb * sampling_ratio_time_domain_keep), Nue, M, N_s, log2(M)
        b, llr = inputs
        b = tf.transpose(b, [0, 1, 3, 2, 4, 5])
        bmi = self.BMI_forall_samples([b, llr])
        return -tf.reduce_mean(bmi)
