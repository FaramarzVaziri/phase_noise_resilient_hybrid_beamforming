
import tensorflow as tf
import numpy as np
class Loss_phase_noise_free():


    def __init__(self, setup):
        super(Loss_phase_noise_free, self).__init__()
        self.setup = setup

    @tf.function
    def C_per_sample_per_k(self,bundeled_inputs):
        V_D_cplx, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, k = bundeled_inputs  # no vectorization
        T0 = tf.linalg.matmul(W_D_cplx[k, :], W_RF_cplx, adjoint_a=True, adjoint_b=True)
        T1 = tf.linalg.matmul(T0, H_complex[k, :], adjoint_a=False, adjoint_b=False)
        T2 = tf.linalg.matmul(T1, V_RF_cplx, adjoint_a=False, adjoint_b=False)
        T3 = tf.linalg.matmul(T2, V_D_cplx[k, :], adjoint_a=False, adjoint_b=False)
        R_X = tf.linalg.matmul(T3, T3, adjoint_a=False, adjoint_b=True)
        R_Q = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)
        T4 = tf.cond(tf.equal(tf.zeros([1], dtype=tf.complex64), tf.linalg.det(R_Q)),
                     lambda: tf.multiply(tf.zeros([1], dtype=tf.complex64), R_Q), lambda: tf.linalg.inv(R_Q))
        # T4 = tf.linalg.inv(R_Q)
        T5 = tf.linalg.matmul(T4, R_X, adjoint_a=False, adjoint_b=False)
        T6 = tf.add(tf.eye(self.setup.N_s, dtype=tf.complex64), tf.divide(T5, tf.cast(self.setup.sigma2, dtype=tf.complex64)))
        T7 = tf.math.real(tf.linalg.det(T6))
        eta = 0.
        # T8 = tf.cond(tf.less(0.0 , T7), lambda: tf.divide(tf.math.log( T7 ) , tf.math.log(2.0)), lambda: tf.multiply(eta , T7))
        T8 = tf.divide(tf.math.log(T7), tf.math.log(2.0))
        return T8


    @tf.function
    def capacity_forall_k(self, bundeled_inputs_0):
        # k_vec = tf.convert_to_tensor(
        #     np.random.choice(self.setup.K, round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K), replace=False),
        #     dtype=tf.int32)


        k_vec = tf.reshape(tf.range(0, self.setup.K_prime, round(1 / self.setup.sampling_ratio_subcarrier_domain_keep), dtype=tf.int32),
                           shape=[round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K_prime)])

        V_D, W_D, H, V_RF, W_RF = bundeled_inputs_0  # [k, ...]
        V_D = tf.tile([V_D], multiples=[round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K_prime), 1, 1, 1])
        W_D = tf.tile([W_D], multiples=[round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K_prime), 1, 1, 1])
        H = tf.tile([H], multiples=[round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K_prime), 1, 1, 1])
        V_RF = tf.tile([V_RF], multiples=[round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K_prime), 1, 1])
        W_RF = tf.tile([W_RF], multiples=[round(self.setup.sampling_ratio_subcarrier_domain_keep * self.setup.K_prime), 1, 1])
        bundeled_inputs_vectorized_on_k = [V_D, W_D, H, V_RF, W_RF, k_vec]
        T0 = tf.map_fn(self.C_per_sample_per_k, bundeled_inputs_vectorized_on_k,
                           fn_output_signature=tf.float32) # , parallel_iterations=self.K
        return tf.reduce_mean(T0)

    @tf.function
    def capacity_forall_symbols(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF = bundeled_inputs_0  # [Nsymb, k, ...]
        H = tf.tile([H], multiples=[round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep), 1, 1, 1])
        bundeled_inputs_1 = [V_D, W_D, H, V_RF, W_RF]
        # print('bundeled_inputs_1 =', bundeled_inputs_1)
        c = tf.reduce_mean(tf.map_fn(self.capacity_forall_k, bundeled_inputs_1,
                                     fn_output_signature=tf.float32,
                                     parallel_iterations = round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep)), axis=0)
        return c

    @tf.function
    def capacity_forall_samples(self, bundeled_inputs):
        if (self.setup.impl == 'map_fn'):
            T0 = tf.map_fn(self.capacity_forall_symbols, bundeled_inputs, fn_output_signature=tf.float32, parallel_iterations=self.setup.BATCHSIZE) #
            return tf.multiply(-1.0, tf.reduce_mean(T0))
