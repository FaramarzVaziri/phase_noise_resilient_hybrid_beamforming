import numpy as np
import scipy.io as sio
import tensorflow as tf


class PhaseNoiseOps():
    def __init__(self, N_a, N_rf):
        self.N_a = N_a
        self.N_rf = N_rf

    @tf.function
    def PHN_forall_RF(self, theta):
        return tf.linalg.diag(tf.repeat(theta, repeats=tf.cast(self.N_a / self.N_rf, dtype=tf.int32), axis=0))

    @tf.function
    def PHN_forall_RF_forall_K(self, inputs):
        return tf.map_fn(self.PHN_forall_RF, inputs, fn_output_signature=tf.complex64)

    @tf.function
    def PHN_forall_RF_forall_K_forall_symbols(self, inputs):
        return tf.map_fn(self.PHN_forall_RF_forall_K, inputs, fn_output_signature=tf.complex64)

    @tf.function
    def PHN_forall_RF_forall_K_forall_symbols_forall_samples(self, inputs):
        return tf.map_fn(self.PHN_forall_RF_forall_K_forall_symbols, inputs, fn_output_signature=tf.complex64)


class PhaseNoiseGen():
    def __init__(self, train_or_test, phase_noise_exists, setup):
        super(PhaseNoiseGen, self).__init__()
        self.train_or_test = train_or_test
        self.phase_noise_exists = phase_noise_exists
        self.setup = setup

    @tf.function
    def Wiener_phase_noise_generator_Ruoyu_for_one_frame_forall_RF(self, Nrf):
        rand_start = 0  # np.random.random_integers(low=0, high=round(1 / self.sampling_ratio_time_domain_keep)-1)
        selected_symbols = range(0 + rand_start,
                                 self.setup.CSIRSPeriod + rand_start - round(
                                     1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                 round(1 / self.setup.sampling_ratio_time_domain_keep))
        if self.setup.CLO_or_ILO == 'ILO':
            DFT_phn_tmp = []
            for nr in range(Nrf):
                T0 = tf.random.normal(shape=[self.setup.CSIRSPeriod * self.setup.K],
                                      mean=0.0,
                                      stddev=self.setup.PHN_innovation_std,
                                      dtype=tf.float32)
                random_offsets_per_frame = tf.random.uniform([1], minval=-np.pi, maxval=np.pi, dtype=tf.dtypes.float32)

                PHN_time = tf.math.cumsum(T0) + tf.tile(random_offsets_per_frame,
                                                        multiples=[self.setup.CSIRSPeriod * self.setup.K])
                PHN_time_reshaped = tf.reshape(PHN_time, shape=[self.setup.CSIRSPeriod, self.setup.K])
                PHN_time_reshaped_sampled = tf.gather(PHN_time_reshaped, selected_symbols, axis=0)
                exp_j_PHN_time_reshaped = tf.complex(tf.cos(PHN_time_reshaped_sampled),
                                                     tf.sin(PHN_time_reshaped_sampled))
                DFT_of_exp_j_PHN_time_reshaped = tf.signal.fft(exp_j_PHN_time_reshaped) / self.setup.K
                DFT_phn_tmp.append(tf.concat([DFT_of_exp_j_PHN_time_reshaped[:, 0: round(self.setup.K_prime / 2)],
                                              DFT_of_exp_j_PHN_time_reshaped[:,
                                              self.setup.K - round(self.setup.K_prime / 2): self.setup.K]],
                                             axis=1))
            T1 = tf.stack(DFT_phn_tmp, axis=0)
            R = tf.transpose(T1, perm=[1, 2, 0])

        elif self.setup.CLO_or_ILO == 'CLO':
            T0 = tf.random.normal(shape=[self.setup.CSIRSPeriod * self.setup.K],
                                  mean=0.0,
                                  stddev=self.setup.PHN_innovation_std,
                                  dtype=tf.float32)
            random_offsets_per_frame = tf.random.uniform([1], minval=-np.pi, maxval=np.pi, dtype=tf.dtypes.float32)

            PHN_time = tf.math.cumsum(T0) + tf.tile(random_offsets_per_frame,
                                                    multiples=[self.setup.CSIRSPeriod * self.setup.K])
            PHN_time_reshaped = tf.reshape(PHN_time, shape=[self.setup.CSIRSPeriod, self.setup.K])
            PHN_time_reshaped_sampled = tf.gather(PHN_time_reshaped, selected_symbols, axis=0)
            exp_j_PHN_time_reshaped = tf.complex(tf.cos(PHN_time_reshaped_sampled),
                                                 tf.sin(PHN_time_reshaped_sampled))
            DFT_of_exp_j_PHN_time_reshaped = tf.signal.fft(exp_j_PHN_time_reshaped) / self.setup.K
            T1 = tf.concat([DFT_of_exp_j_PHN_time_reshaped[:, 0: round(self.setup.K_prime / 2)],
                            DFT_of_exp_j_PHN_time_reshaped[:,
                            self.setup.K - round(self.setup.K_prime / 2): self.setup.K]],
                           axis=1)
            T2 = tf.tile(tf.expand_dims(T1, axis=0), multiples=[Nrf, 1, 1])
            R = tf.transpose(T2, perm=[1, 2, 0])
        return R

    @tf.function
    def PHN_for_entire_batch(self, Nrf):
        DFT_of_exp_of_jPHN_tmp = []
        for ij in range(self.setup.BATCHSIZE):
            DFT_of_exp_of_jPHN_tmp.append(self.Wiener_phase_noise_generator_Ruoyu_for_one_frame_forall_RF(Nrf))
        return tf.stack(DFT_of_exp_of_jPHN_tmp, axis=0)

    @tf.function
    def phase_noise_dataset_generator(self):
        # BS
        PHN_B_DFT_domain_samples_K_Nrf_train = self.PHN_for_entire_batch(self.setup.N_b_rf)
        phi_ops_BS = PhaseNoiseOps(self.setup.N_b_a, self.setup.N_b_rf)
        Lambda_B = phi_ops_BS.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_B_DFT_domain_samples_K_Nrf_train)
        # UE
        phi_ops_UE = PhaseNoiseOps(self.setup.N_u_a, self.setup.N_u_rf)
        Lambda_U_tmp = []
        for u in range(self.setup.Nue):
            PHN_U_DFT_domain_samples_K_Nrf_train = self.PHN_for_entire_batch(self.setup.N_u_rf)
            Lambda_U_tmp.append(
                phi_ops_UE.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_U_DFT_domain_samples_K_Nrf_train))
        Lambda_U = tf.stack(Lambda_U_tmp, axis=2)

        return Lambda_B, Lambda_U

    @tf.function
    def phase_noise_record_player(self, PHN_time):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # phi [self.setup.N_rf, self.setup.CSIRSPeriod * self.setup.K]
        rand_start = 0  # np.random.random_integers(low=0, high=round(1 / self.sampling_ratio_time_domain_keep)-1)
        selected_symbols = range(0 + rand_start,
                                 self.setup.CSIRSPeriod + rand_start - round(
                                     1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                 round(1 / self.setup.sampling_ratio_time_domain_keep))
        PHN_time_reshaped = tf.reshape(PHN_time, shape=[-1, self.setup.CSIRSPeriod, self.setup.K])
        PHN_time_reshaped_sampled = tf.gather(PHN_time_reshaped, selected_symbols, axis=1)
        exp_j_PHN_time_reshaped = tf.complex(tf.cos(PHN_time_reshaped_sampled),
                                             tf.sin(PHN_time_reshaped_sampled))
        DFT_of_exp_j_PHN_time_reshaped = tf.signal.fft(exp_j_PHN_time_reshaped) / self.setup.K
        DFT_phn_tmp_1 = tf.concat([DFT_of_exp_j_PHN_time_reshaped[:, :, 0: round(self.setup.K_prime / 2)],
                                   DFT_of_exp_j_PHN_time_reshaped[:, :,
                                   self.setup.K - round(self.setup.K_prime / 2): self.setup.K]],
                                  axis=2)
        return tf.transpose(DFT_phn_tmp_1, perm=[1, 2, 0])

    @tf.function
    def phase_noise_record_player_forall_samples(self, inputs):
        # HOW THE INPUTS COME IN --------------------------------------------------
        # inputs [self.setup.BATCHSIZE, self.setup.N_rf, self.setup.CSIRSPeriod * self.setup.K]
        return tf.map_fn(self.phase_noise_record_player, inputs, fn_output_signature=tf.complex64)

    @tf.function
    def phase_noise_dataset_reader(self, phi):
        phi = tf.reshape(phi, [self.setup.BATCHSIZE, (self.setup.N_b_rf + self.setup.Nue * self.setup.N_u_rf),
                               self.setup.CSIRSPeriod * self.setup.K])
        phi_BS = phi[:, 0:self.setup.N_b_rf, :]
        PHN_B_DFT_domain_samples_K_Nrf_train = self.phase_noise_record_player_forall_samples(phi_BS)
        phi_ops_BS = PhaseNoiseOps(self.setup.N_b_a, self.setup.N_b_rf)
        Lambda_B = phi_ops_BS.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_B_DFT_domain_samples_K_Nrf_train)

        phi_UE = tf.reshape(phi[:, self.setup.N_b_rf: self.setup.N_b_rf + self.setup.Nue * self.setup.N_u_rf, :],
                            [self.setup.Nue, self.setup.BATCHSIZE, self.setup.N_u_rf,
                             self.setup.CSIRSPeriod * self.setup.K])
        # UE
        phi_ops_UE = PhaseNoiseOps(self.setup.N_u_a, self.setup.N_u_rf)
        Lambda_U_tmp = []
        for u in range(self.setup.Nue):
            PHN_U_DFT_domain_samples_K_Nrf_train = self.phase_noise_record_player_forall_samples(phi_UE[u, :, :, :])
            Lambda_U_tmp.append(
                phi_ops_UE.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_U_DFT_domain_samples_K_Nrf_train))
        Lambda_U = tf.stack(Lambda_U_tmp, axis=2)
        return Lambda_B, Lambda_U


class ChannelOps(PhaseNoiseGen):
    def __init__(self, train_or_test, phase_noise_exists, setup):
        super(ChannelOps, self).__init__(train_or_test, phase_noise_exists, setup)
        self.train_or_test = train_or_test
        self.phase_noise_exists = phase_noise_exists

    @tf.function
    def cyclical_shift(self, Lambda_matrix, k, flip):
        return tf.cond(tf.equal(flip, True),
                       lambda: tf.roll(tf.reverse(Lambda_matrix, axis=[0]), shift=tf.squeeze(k) + 1, axis=0),
                       lambda: tf.roll(Lambda_matrix, shift=tf.squeeze(k), axis=0))

    @tf.function
    def h_tilde_k_q(self, inputs):
        H, Lambda_B_cyclshifted, Lambda_U_cyclshifted = inputs
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted,
                                                 H), Lambda_B_cyclshifted)

    @tf.function
    def h_tilde_k(self, inputs):
        H, Lambda_B, Lambda_U, k = inputs
        inputs1 = [H, self.cyclical_shift(Lambda_B, k, False), self.cyclical_shift(Lambda_U, k, True)]
        return tf.reduce_sum(tf.map_fn(self.h_tilde_k_q, inputs1, fn_output_signature=tf.complex64),
                             axis=0)

    @tf.function
    def h_tilde_forall_k(self, bundeled_inputs_0):
        H, Lambda_B, Lambda_U = bundeled_inputs_0
        # repeating for function vectorization
        all_k = tf.reshape(tf.range(0, self.setup.K_prime, 1), shape=[self.setup.K_prime, 1])
        H_repeated_K_times = tf.tile([H], multiples=[self.setup.K_prime, 1, 1, 1])
        Lambda_B_repeated_K_times = tf.tile([Lambda_B], multiples=[self.setup.K_prime, 1, 1, 1])
        Lambda_U_repeated_K_times = tf.tile([Lambda_U], multiples=[self.setup.K_prime, 1, 1, 1])

        bundeled_inputs_1 = [H_repeated_K_times, Lambda_B_repeated_K_times,
                             Lambda_U_repeated_K_times, all_k]
        H_tilde_complex = tf.map_fn(self.h_tilde_k, bundeled_inputs_1,
                                    fn_output_signature=tf.complex64,
                                    parallel_iterations=self.setup.K_prime)  # parallel over all k subcarriers
        return H_tilde_complex

    @tf.function
    def h_tilde_forall_u(self, inputs):
        H, Lambda_B, Lambda_U = inputs
        # repeating for function vectorization
        Lambda_B_repeated_Nue_times = tf.tile([Lambda_B], multiples=[self.setup.Nue, 1, 1, 1])
        inputs_1 = [H, Lambda_B_repeated_Nue_times, Lambda_U]
        return tf.map_fn(self.h_tilde_forall_k, inputs_1,
                         fn_output_signature=tf.complex64,
                         parallel_iterations=self.setup.Nue)

    @tf.function
    def h_tilde_forall_ns(self, bundeled_inputs_0):  # Nsymb, K, ...
        H, Lambda_B, Lambda_U = bundeled_inputs_0
        H_repeated_Nsymb_times = tf.tile([H], multiples=[
            round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep), 1, 1, 1, 1])
        bundeled_inputs_1 = [H_repeated_Nsymb_times, Lambda_B, Lambda_U]
        H_tilde_complex = tf.map_fn(self.h_tilde_forall_u, bundeled_inputs_1,
                                    fn_output_signature=tf.complex64,
                                    parallel_iterations=round(
                                        self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep))
        return H_tilde_complex


class DatasetGenerator(ChannelOps, PhaseNoiseGen):

    def __init__(self, train_or_test, phase_noise_exists, data_fragment_size,
                 dataset_id_start, evaluation_bits, evaluation_symbols, setup):
        super(DatasetGenerator, self).__init__(train_or_test, phase_noise_exists, setup)
        self.train_or_test = train_or_test
        self.phase_noise_exists = phase_noise_exists
        self.data_fragment_size = data_fragment_size
        self.dataset_id_start = dataset_id_start
        self.evaluation_bits = evaluation_bits
        self.evaluation_symbols = evaluation_symbols

    @tf.function
    def segmented_channel_dataset_generator(self, mat_fname):
        mat_contents = sio.loadmat(mat_fname)
        H = np.zeros(
            shape=[self.data_fragment_size, self.setup.Nue, self.setup.K_prime, self.setup.N_u_a, self.setup.N_b_a, 2],
            dtype=np.float32)
        if (self.train_or_test == "train"):
            var_name_real = "H_real_" + "train"
            H[:, :, :, :, :, 0] = np.transpose(mat_contents[var_name_real], axes=[0, 1, 4, 2, 3])[
                                  0:self.data_fragment_size,
                                  :, :,
                                  :]
            var_name_imag = "H_imag_" + "train"
            H[:, :, :, :, :, 1] = np.transpose(mat_contents[var_name_imag], axes=[0, 1, 4, 2, 3])[
                                  0:self.data_fragment_size,
                                  :, :,
                                  :]
        else:
            var_name_real = "H_real_" + "test"
            H[:, :, :, :, :, 0] = \
                np.transpose(mat_contents[var_name_real], axes=[0, 1, 4, 2, 3])[0:self.data_fragment_size, :, :, :]
            var_name_imag = "H_imag_" + "test"
            H[:, :, :, :, :, 1] = \
                np.transpose(mat_contents[var_name_imag], axes=[0, 1, 4, 2, 3])[0:self.data_fragment_size, :, :, :]

        H_complex = tf.complex(H[:, :, :, :, :, 0], H[:, :, :, :, :, 1])
        DS = tf.data.Dataset.from_tensor_slices(H_complex)
        return DS

    @tf.function
    def segmented_phase_noise_dataset_generator(self, mat_fname):
        mat_contents = sio.loadmat(mat_fname)
        var_name = "PNsamps"
        phi = mat_contents[var_name]
        required_len = self.data_fragment_size * (
                self.setup.N_b_rf + self.setup.Nue * self.setup.N_u_rf) * self.setup.CSIRSPeriod * self.setup.K
        if required_len < len(phi):
            phi_adjusted = phi[0:required_len]
        else:
            # phi_adjusted = phi
            T = []
            for i in range(round(required_len / len(phi)) + 1):
                T.append(i * (phi[-1, :] - phi[0, :]))
            adjuster = tf.stack(T, axis=0)
            adjuster_repeated = tf.repeat(adjuster, repeats=[len(phi)], axis=0)
            adjuster_repeated = adjuster_repeated[0:required_len, :]

            phi_1 = tf.concat([phi, tf.tile(phi, multiples=[round(required_len / len(phi)), 1])], axis=0)
            phi_adjusted = phi_1[0:required_len, :] + adjuster_repeated

            phi_adjusted = tf.reshape(phi_adjusted, shape=[self.data_fragment_size,
                                                           (
                                                                       self.setup.N_b_rf + self.setup.Nue * self.setup.N_u_rf) * self.setup.CSIRSPeriod * self.setup.K])
        DS = tf.data.Dataset.from_tensor_slices(phi_adjusted)
        return DS

    @tf.function
    def dataset_generator(self):
        DS_channel = self.segmented_channel_dataset_generator(
            self.setup.dataset_name)  # if small sys, DS is loaded at once
        if (self.setup.phase_noise_recorded == 'yes'):
            DS_phi = self.segmented_phase_noise_dataset_generator(self.setup.dataset_phi_address)
            DS = tf.data.Dataset.zip((DS_channel, DS_phi))
        else:
            DS = DS_channel

        DS = DS.cache()
        DS = DS.batch(self.setup.BATCHSIZE)
        DS = DS.map(self.dataset_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        DS = DS.prefetch(AUTOTUNE)

        return DS

    @tf.function
    def dataset_mapper(self, *inputs):  # batch, Nsymb, K, ...
        if (self.setup.phase_noise_recorded == 'yes'):
            H_complex, phi = inputs
            Lambda_B, Lambda_U = self.phase_noise_dataset_reader(phi)
        else:
            H_complex = inputs[0]
            Lambda_B, Lambda_U = self.phase_noise_dataset_generator()

        bundeled_inputs_0 = [H_complex, Lambda_B, Lambda_U]
        H_tilde_complex = tf.map_fn(self.h_tilde_forall_ns, bundeled_inputs_0, fn_output_signature=tf.complex64,
                                    parallel_iterations=self.setup.BATCHSIZE)
        H_tilde = tf.stack([tf.math.real(H_tilde_complex), tf.math.imag(H_tilde_complex)], axis=6)
        H = tf.stack([tf.math.real(H_complex), tf.math.imag(H_complex)], axis=5)

        return H, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U

    # @tf.function
    def data_generator_for_evaluation_of_proposed_beamformer(self, batch_number):
        mat_contents = sio.loadmat(self.setup.dataset_name)
        # print('# mat_contents', mat_contents)
        H = np.zeros(
            shape=[self.setup.BATCHSIZE, self.setup.Nue, self.setup.K_prime, self.setup.N_u_a, self.setup.N_b_a, 2],
            dtype=np.float32)
        # print('# H', H)
        var_name_real = "H_real_" + "test"
        H[:, :, :, :, :, 0] = \
            np.transpose(mat_contents[var_name_real], axes=[0, 1, 4, 2, 3])[
            batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :]
        var_name_imag = "H_imag_" + "test"
        H[:, :, :, :, :, 1] = \
            np.transpose(mat_contents[var_name_imag], axes=[0, 1, 4, 2, 3])[
            batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :]
        H_complex = tf.complex(H[:, :, :, :, :, 0], H[:, :, :, :, :, 1])
        Lambda_B, Lambda_U = self.phase_noise_dataset_generator()
        bundeled_inputs_0 = [H_complex, Lambda_B, Lambda_U]

        H_tilde_complex = tf.map_fn(self.h_tilde_forall_ns, bundeled_inputs_0, fn_output_signature=tf.complex64,
                                    parallel_iterations=self.setup.BATCHSIZE)
        H_tilde = tf.stack([tf.math.real(H_tilde_complex), tf.math.imag(H_tilde_complex)], axis=6)
        return H_complex, H_tilde, Lambda_B, Lambda_U


    def data_generator_for_running_Sohrabis_beamformer(self):
        H_complex, H_tilde, Lambda_B, Lambda_U = self.data_generator_for_evaluation_of_proposed_beamformer(
            0)  # todo: this is why test dataset should remain the same length as BATCHSIZE. To make it capable of using smaller batchsizes we need to change it to a for loop like other cases, instead of (0)
        H_tilde_complex = tf.complex(H_tilde[:, :, :, :, :, :, 0], H_tilde[:, :, :, :, :, :, 1])
        csi_tx = tf.squeeze(H_tilde_complex[:, 0, :, :, :, :])
        return H_complex, csi_tx, H_tilde, Lambda_B, Lambda_U


    def data_generator_for_evaluation_of_Sohrabis_beamformer(self, batch_number):
        mat_contents = sio.loadmat(self.setup.dataset_for_testing_sohrabi)

        H_complex = (mat_contents['H'])[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE,
                    :, :, :, :]
        H_tilde = (mat_contents['H_tilde'])[
                  batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :,
                  :]
        Lambda_B = (mat_contents['Lambda_B'])[
                   batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :,
                   :, :]
        Lambda_U = (mat_contents['Lambda_U'])[
                   batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :,
                   :,
                   :, :]

        V_RF_Sohrabi_optimized = (mat_contents['V_RF_Sohrabi_optimized'])[
                                 batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :]
        W_RF_Sohrabi_optimized = (mat_contents['W_RF_Sohrabi_optimized'])[
                                 batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :]

        # The following data require permutation to bring k (subcarrier) to the second dimension
        V_D_Sohrabi_optimized = np.transpose(mat_contents['V_D_Sohrabi_optimized'], axes=[0, 3, 1, 2])[
                                batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :]
        W_D_Sohrabi_optimized = np.transpose(mat_contents['W_D_Sohrabi_optimized'], axes=[0, 3, 1, 2])[
                                batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :]
        return tf.cast(H_complex,
                       tf.complex64), H_tilde, Lambda_B, Lambda_U, V_RF_Sohrabi_optimized, \
               tf.reshape(W_RF_Sohrabi_optimized, [-1, self.setup.Nue, self.setup.N_u_a, self.setup.N_u_rf]), \
               tf.reshape(V_D_Sohrabi_optimized,
                          [-1, self.setup.Nue, self.setup.K_prime, self.setup.N_b_rf, self.setup.N_s]) \
            , tf.reshape(W_D_Sohrabi_optimized, [-1, self.setup.Nue, self.setup.K_prime, self.setup.N_u_rf,
                                                 self.setup.N_s])  # , tx_bits, tx_symbols

    # @tf.function
    def data_generator_for_evaluation_of_digital_beamformer(self, batch_number):
        mat_contents = sio.loadmat(self.setup.dataset_for_testing_DBF)
        H_complex = (mat_contents['H'])[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE,
                    :, :, :, :]
        H_tilde = (mat_contents['H_tilde'])[
                  batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :,
                  :]
        Lambda_B = (mat_contents['Lambda_B'])[
                   batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :,
                   :, :]
        Lambda_U = (mat_contents['Lambda_U'])[
                   batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :,
                   :,
                   :, :]
        # Lambda_B, Lambda_U = self.phase_noise_dataset_generator()
        V_RF_optimized = (mat_contents['V_RF_optimized'])[
                                 batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :]
        W_RF_optimized = (mat_contents['W_RF_optimized'])[
                                 batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :]

        # The following data require permutation to bring k (subcarrier) to the second dimension
        V_D_optimized = np.transpose(mat_contents['V_D_optimized'], axes=[0, 3, 1, 2])[
                                batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :]
        W_D_optimized = np.transpose(mat_contents['W_D_optimized'], axes=[0, 3, 1, 2])[
                                batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, :]

        return tf.cast(H_complex,
                       tf.complex64), H_tilde, Lambda_B, Lambda_U, V_RF_optimized, \
               tf.reshape(W_RF_optimized, [-1, self.setup.Nue, self.setup.N_u_a, self.setup.N_u_a]), \
               tf.reshape(V_D_optimized, [-1, self.setup.Nue, self.setup.K_prime, self.setup.N_b_a, self.setup.N_s]) \
            , tf.reshape(W_D_optimized, [-1, self.setup.Nue, self.setup.K_prime, self.setup.N_u_a, self.setup.N_s])