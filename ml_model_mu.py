import numpy as np
import tensorflow as tf

loss_metric = tf.keras.metrics.Mean(name='L2_train')
loss_metric_test = tf.keras.metrics.Mean(name='L2_test')
norm_records = tf.keras.metrics.Mean(name='norm')
capacity_metric_test = tf.keras.metrics.Mean(name='AIR_test')


class ML_model_class(tf.keras.Model):

    def __init__(self, CNN_transceiver, obj_test_dataset, tx_bits_train,
                 tx_symbols_train, demapper, equalizer, setup):
        super(ML_model_class, self).__init__()
        self.CNN_transceiver = CNN_transceiver
        self.obj_test_dataset = obj_test_dataset
        self.tx_bits_train = tx_bits_train
        self.tx_symbols_train = tx_symbols_train
        self.demapper = demapper
        self.equalizer = equalizer
        self.setup = setup
        self.rescaling = 1#self.setup.Nue * self.setup.N_s * np.sqrt(2.)

    def compile(self, optimizer, loss, rx_calc, activation_TX, activation_RX,
                bmi_in_presence_of_phase_noise, capacity_in_presence_of_phase_noise):
        super(ML_model_class, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.rx_calc = rx_calc
        self.bmi_in_presence_of_phase_noise = bmi_in_presence_of_phase_noise
        self.capacity_in_presence_of_phase_noise = capacity_in_presence_of_phase_noise
        self.activation_TX = activation_TX
        self.activation_RX = activation_RX

    @tf.function
    def NN_input_preparation(self, H_tilde):
        csi_rx = H_tilde
        csi_tx_tmp_0 = []
        for u in range(self.setup.Nue):
            csi_tx_tmp_0.append(H_tilde[:, :, u, :, :, :, :])
        csi_tx_tmp_1 = tf.concat(csi_tx_tmp_0, axis=3)  # B, ns, K, Na*Nue, Nb, 2
        if self.setup.K_prime_size_test == 'yes':
            T = csi_tx_tmp_1[:, 0, 0:self.setup.influencial_subcarriers_set_size, :, :, :]
            csi_tx = tf.tile(T,
                             multiples=[1, round(self.setup.K_prime / self.setup.influencial_subcarriers_set_size),1,1,1])
        else:
            csi_tx = csi_tx_tmp_1[:, 0, :, :, :, :]
        return csi_tx, csi_rx

    @tf.function
    def train_step(self, inputs0):
        _, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U = inputs0
        csi_tx, csi_rx = self.NN_input_preparation(H_tilde)

        with tf.GradientTape() as tape:
            V_D_tmp = []
            V_RF_tmp = []
            W_D_tmp = []
            W_RF_tmp = []

            # selected_symbols = np.random.choice(self.Nsymb,
            #                                     round(self.sampling_ratio_time_domain_keep * self.Nsymb),
            #                                     replace=False)
            rand_start = 0  # np.random.random_integers(low=0, high=round(1 / self.sampling_ratio_time_domain_keep)-1)
            selected_symbols = range(0 + rand_start,
                                     self.setup.CSIRSPeriod + rand_start - round(
                                         1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                     round(1 / self.setup.sampling_ratio_time_domain_keep))

            for ns in range(len(selected_symbols)):
                V_D, V_RF, W_D, W_RF = self.CNN_transceiver([csi_tx, tf.tile([ns], multiples=[self.setup.BATCHSIZE])])
                V_D, V_RF = self.activation_TX([V_D, V_RF])
                W_D, W_RF = self.activation_RX([W_D, W_RF])

                V_D_tmp.append(V_D)
                V_RF_tmp.append(V_RF)
                W_D_tmp.append(W_D)
                W_RF_tmp.append(W_RF)

            V_D = tf.stack(V_D_tmp, axis=1)  # [should stack on axis ns]
            V_RF = tf.stack(V_RF_tmp, axis=1)  # [should stack on axis ns]

            W_D = tf.stack(W_D_tmp, axis=1)  # [should stack on axis ns]
            W_RF = tf.stack(W_RF_tmp, axis=1)  # [should stack on axis ns]

            inputs2 = [V_D, W_D, H_complex, V_RF, W_RF, Lambda_B, Lambda_U, self.tx_bits_train, self.tx_symbols_train]

            y_symbols_k0 = self.rx_calc(inputs2)

            inputs3 = [V_D, W_D, tf.complex(csi_rx[:, :, :, :, :, :, 0], csi_rx[:, :, :, :, :, :, 1]), V_RF, W_RF,
                       y_symbols_k0]
            y_symbols_k0_equalized, noise_effective_k0 = self.equalizer(inputs3)
            noise_effective_k0_reshaped = tf.expand_dims(noise_effective_k0, axis=5)
            llr_k0 = self.demapper([y_symbols_k0_equalized / self.rescaling,
                                    noise_effective_k0_reshaped])
            bce_loss_k0 = self.loss([self.tx_bits_train[:, :, :, 0, :, :, :], llr_k0])
        trainables = self.CNN_transceiver.trainable_weights

        grads = tape.gradient(bce_loss_k0, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        loss_metric.update_state(bce_loss_k0)
        return {"L2_train": loss_metric.result()}

    # see https://keras.io/api/models/model_training_apis/ for validation
    @tf.function
    def test_step(self, inputs0):
        _, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U = inputs0

        csi_tx, csi_rx = self.NN_input_preparation(H_tilde)
        # selected_symbols = np.random.choice(self.Nsymb,
        #                                         round(self.sampling_ratio_time_domain_keep * self.Nsymb),
        #                                         replace=False)
        rand_start = 0  # np.random.random_integers(low=0, high=round(1 / self.sampling_ratio_time_domain_keep)-1)
        selected_symbols = range(0 + rand_start,
                                 self.setup.CSIRSPeriod + rand_start - round(
                                     1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                 round(1 / self.setup.sampling_ratio_time_domain_keep))


        V_D_tmp = []
        V_RF_tmp = []
        W_D_tmp = []
        W_RF_tmp = []

        the_mask_of_ns = np.zeros(shape=self.setup.CSIRSPeriod, dtype=np.int32)
        for ns in range(len(selected_symbols)):
            V_D, V_RF, W_D, W_RF = self.CNN_transceiver([csi_tx, tf.tile([ns], multiples=[self.setup.BATCHSIZE])])
            V_D, V_RF = self.activation_TX([V_D, V_RF])
            W_D, W_RF = self.activation_RX([W_D, W_RF])
            V_D_tmp.append(V_D)
            V_RF_tmp.append(V_RF)
            W_D_tmp.append(W_D)
            W_RF_tmp.append(W_RF)

        V_D = tf.stack(V_D_tmp, axis=1)  # [should stack on axis ns]
        V_RF = tf.stack(V_RF_tmp, axis=1)  # [should stack on axis ns]
        W_D = tf.stack(W_D_tmp, axis=1)  # [should stack on axis ns]
        W_RF = tf.stack(W_RF_tmp, axis=1)  # [should stack on axis ns]

        inputs2 = [V_D, W_D, H_complex, V_RF, W_RF, Lambda_B, Lambda_U, self.tx_bits_train, self.tx_symbols_train]
        y_symbols_k0 = self.rx_calc(inputs2)

        inputs3 = [V_D, W_D, tf.complex(csi_rx[:, :, :, :, :, :, 0], csi_rx[:, :, :, :, :, :, 1]), V_RF, W_RF,
                   y_symbols_k0]
        y_symbols_k0_equalized, noise_effective_k0 = self.equalizer(inputs3)
        noise_effective_k0_reshaped = tf.expand_dims(noise_effective_k0, axis=5)
        llr_k0 = self.demapper([y_symbols_k0_equalized / self.rescaling, noise_effective_k0_reshaped])
        bce_loss = self.loss([self.tx_bits_train[:, :, :, 0, :, :, :], llr_k0])
        loss_metric_test.update_state(bce_loss)


        # selected_symbols_capacity_metric = range(0, self.Nsymb - round(
        #     1 / self.sampling_ratio_time_domain_keep_capacity_metric) + 1,
        #                                          round(1 / self.sampling_ratio_time_domain_keep_capacity_metric))
        # selected_symbols_capacity_metric = [24]
        V_D_tmp_ = []
        V_RF_tmp_ = []
        W_D_tmp_ = []
        W_RF_tmp_ = []

        for ns in range(len(selected_symbols)):
            V_D_, V_RF_, W_D_, W_RF_ = self.CNN_transceiver([csi_tx, tf.tile([ns], multiples=[self.setup.BATCHSIZE])])
            V_D_, V_RF_ = self.activation_TX([V_D_, V_RF_])
            W_D_, W_RF_ = self.activation_RX([W_D_, W_RF_])
            V_D_tmp_.append(V_D_)
            V_RF_tmp_.append(V_RF_)
            W_D_tmp_.append(W_D_)
            W_RF_tmp_.append(W_RF_)

        V_D_ = tf.stack(V_D_tmp_, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
        V_RF_ = tf.stack(V_RF_tmp_, axis=1)  # batch, Nsymb, ... [should stack on axis ns]
        W_D_ = tf.stack(W_D_tmp_, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
        W_RF_ = tf.stack(W_RF_tmp_, axis=1)  # batch, Nsymb, ... [should stack on axis ns]

        inputs4 = [V_D_, W_D_, H_complex, V_RF_, W_RF_, Lambda_B, Lambda_U, self.tx_bits_train, self.tx_symbols_train]
        y_symbols_k0_ = self.rx_calc(inputs4)

        inputs6 = [V_D_, W_D_, tf.complex(csi_rx[:, :, :, :, :, :, 0], csi_rx[:, :, :, :, :, :, 1]), V_RF_, W_RF_,
                   y_symbols_k0_]
        y_symbols_k0_equalized_, noise_effective_k0_ = self.equalizer(inputs6)
        noise_effective_k0_reshaped = tf.expand_dims(noise_effective_k0_, axis=5)
        llr__k0 = self.demapper(
            [y_symbols_k0_equalized_ / self.rescaling, noise_effective_k0_reshaped])  # self.sigma2 / (2 * np.pi)])
        bmi_avg, _ = self.bmi_in_presence_of_phase_noise([self.tx_bits_train[:, :, :, 0, :, :, :], llr__k0])
        capacity_metric_test.update_state(bmi_avg)
        return {"L2_test": loss_metric_test.result(),
                'AIR_test': capacity_metric_test.result()}

    @tf.function
    def evaluation_of_proposed_beamformer(self, tx_bits, tx_symbols):

        # selected_symbols = np.random.choice(self.Nsymb,
        #                                         round(self.sampling_ratio_time_domain_keep * self.Nsymb),
        #                                         replace=False)
        rand_start = 0
        selected_symbols = range(0 + rand_start,
                                 self.setup.CSIRSPeriod + rand_start - round(
                                     1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                 round(1 / self.setup.sampling_ratio_time_domain_keep))


        N_of_batches_in_DS = round(self.setup.eval_dataset_size / self.setup.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            H_complex, H_tilde, Lambda_B, Lambda_U = \
                self.obj_test_dataset.data_generator_for_evaluation_of_proposed_beamformer(batch_number)

            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)

            V_D_tmp = []
            V_RF_tmp = []
            W_D_tmp = []
            W_RF_tmp = []

            for ns in range(len(selected_symbols)):
                V_D, V_RF, W_D, W_RF = self.CNN_transceiver([csi_tx, tf.tile([ns], multiples=[self.setup.BATCHSIZE])])
                V_D, V_RF = self.activation_TX([V_D, V_RF])
                W_D, W_RF = self.activation_RX([W_D, W_RF])
                V_D_tmp.append(V_D)
                V_RF_tmp.append(V_RF)
                W_D_tmp.append(W_D)
                W_RF_tmp.append(W_RF)

            V_D = tf.stack(V_D_tmp, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            V_RF = tf.stack(V_RF_tmp, axis=1)  # batch, Nsymb, ... [should stack on axis ns]

            W_D = tf.stack(W_D_tmp, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            W_RF = tf.stack(W_RF_tmp, axis=1)  # batch, Nsymb, ... [should stack on axis ns]

            inputs0 = [V_D, W_D, H_complex, V_RF, W_RF, Lambda_B, Lambda_U,
                       tx_bits[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :],
                       tx_symbols[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :]]
            y_symbols_k0 = self.rx_calc(inputs0)

            inputs3 = [V_D, W_D, tf.complex(csi_rx[:, :, :, :, :, :, 0], csi_rx[:, :, :, :, :, :, 1]), V_RF, W_RF,
                       y_symbols_k0]
            y_symbols_k0_equalized, noise_effective_k0 = self.equalizer(inputs3)
            noise_effective_k0_reshaped = tf.expand_dims(noise_effective_k0, axis=5)
            llr_k0 = self.demapper(
                [y_symbols_k0_equalized / self.rescaling, noise_effective_k0_reshaped])  # self.sigma2 / (2 * np.pi)])

            inputs1 = [
                tx_bits[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, 0, :, :,
                :], llr_k0]

            bmi_avg, bmi = self.bmi_in_presence_of_phase_noise(inputs1)

            if (batch_number == 0):
                air_samples_k0 = bmi
            else:
                air_samples_k0 = tf.concat([air_samples_k0, bmi], axis=0)

        return air_samples_k0, y_symbols_k0_equalized / self.rescaling

    @tf.function
    def evaluation_of_Sohrabis_beamformer(self, tx_bits, tx_symbols):
        # selected_symbols = np.random.choice(self.Nsymb,
        #                                         round(self.sampling_ratio_time_domain_keep * self.Nsymb),
        #                                         replace=False)
        rand_start = 0
        selected_symbols_capacity_metric = range(0 + rand_start,
                                                 self.setup.CSIRSPeriod + rand_start - round(
                                                     1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                                 round(1 / self.setup.sampling_ratio_time_domain_keep))

        N_of_batches_in_DS = round(self.setup.eval_dataset_size / self.setup.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            # print('batch_number: ', batch_number)
            H_complex, H_tilde, Lambda_B, Lambda_U, V_RF_Sohrabi_optimized, W_RF_Sohrabi_optimized, \
            V_D_Sohrabi_optimized, W_D_Sohrabi_optimized = \
                self.obj_test_dataset.data_generator_for_evaluation_of_Sohrabis_beamformer(batch_number)

            V_D_Sohrabi_optimized = tf.tile(tf.expand_dims(V_D_Sohrabi_optimized, axis=1), multiples=[1,
                                                                                                      round(
                                                                                                          self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                                      1, 1, 1, 1])
            W_D_Sohrabi_optimized = tf.tile(tf.expand_dims(W_D_Sohrabi_optimized, axis=1), multiples=[1,
                                                                                                      round(
                                                                                                          self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                                      1, 1, 1, 1])
            V_RF_Sohrabi_optimized = tf.tile(tf.expand_dims(V_RF_Sohrabi_optimized, axis=1), multiples=[1,
                                                                                                        round(
                                                                                                            self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                                        1, 1])
            W_RF_Sohrabi_optimized = tf.tile(tf.expand_dims(W_RF_Sohrabi_optimized, axis=1), multiples=[1,
                                                                                                        round(
                                                                                                            self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                                        1, 1, 1])

            inputs0 = [V_D_Sohrabi_optimized,
                       W_D_Sohrabi_optimized,
                       H_complex,
                       V_RF_Sohrabi_optimized,
                       W_RF_Sohrabi_optimized,
                       Lambda_B,
                       Lambda_U,
                       tx_bits[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :],
                       tx_symbols[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :]]
            y_symbols_k0 = self.rx_calc(inputs0)

            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)

            inputs3 = [V_D_Sohrabi_optimized, W_D_Sohrabi_optimized,
                       tf.complex(csi_rx[:, :, :, :, :, :, 0], csi_rx[:, :, :, :, :, :, 1]),
                       V_RF_Sohrabi_optimized, W_RF_Sohrabi_optimized, y_symbols_k0]
            y_symbols_k0_equalized, noise_effective_k0 = self.equalizer(inputs3)
            noise_effective_k0_reshaped = tf.expand_dims(noise_effective_k0, axis=5)
            llr_k0 = self.demapper(
                [y_symbols_k0_equalized / self.rescaling, noise_effective_k0_reshaped])  # self.setup.sigma2 / (2 * np.pi)

            # BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, M 2, K_prime 3, Nue 4, N_s 5, log2(M) 6
            inputs1 = [
                tx_bits[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, 0, :, :,
                :],
                llr_k0]  # BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, M 2, Nue 3, N_s 4, log2(M) 5

            bmi_avg, bmi = self.bmi_in_presence_of_phase_noise(inputs1)

            if (batch_number == 0):
                air_samples_k0 = bmi
            else:
                air_samples_k0 = tf.concat([air_samples_k0, bmi], axis=0)

        return air_samples_k0, y_symbols_k0_equalized / self.rescaling

    @tf.function
    def evaluation_of_digital_beamformer(self, tx_bits, tx_symbols):
        # selected_symbols = np.random.choice(self.Nsymb,
        #                                         round(self.sampling_ratio_time_domain_keep * self.Nsymb),
        #                                         replace=False)
        rand_start = 0
        selected_symbols_capacity_metric = range(0 + rand_start,
                                                 self.setup.CSIRSPeriod + rand_start - round(
                                                     1 / self.setup.sampling_ratio_time_domain_keep) + 1,
                                                 round(1 / self.setup.sampling_ratio_time_domain_keep))

        N_of_batches_in_DS = round(self.setup.eval_dataset_size / self.setup.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            # print('batch_number: ', batch_number)
            H_complex, H_tilde, Lambda_B, Lambda_U, V_RF_optimized, W_RF_optimized, \
            V_D_optimized, W_D_optimized = \
                self.obj_test_dataset.data_generator_for_evaluation_of_digital_beamformer(batch_number)

            V_D_optimized = tf.tile(tf.expand_dims(V_D_optimized, axis=1), multiples=[1, round(
                self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                      1, 1, 1, 1])
            W_D_optimized = tf.tile(tf.expand_dims(W_D_optimized, axis=1), multiples=[1,
                                                                                      round(
                                                                                          self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                      1, 1, 1, 1])
            V_RF_optimized = tf.tile(tf.expand_dims(V_RF_optimized, axis=1), multiples=[1,
                                                                                        round(
                                                                                            self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                        1, 1])
            W_RF_optimized = tf.tile(tf.expand_dims(W_RF_optimized, axis=1), multiples=[1,
                                                                                        round(
                                                                                            self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep_capacity_metric),
                                                                                        1, 1, 1])

            inputs0 = [V_D_optimized,
                       W_D_optimized,
                       H_complex,
                       V_RF_optimized,
                       W_RF_optimized,
                       Lambda_B,
                       Lambda_U,
                       tx_bits[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :],
                       tx_symbols[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :]]
            y_symbols_k0 = self.rx_calc(inputs0)


            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)

            inputs3 = [V_D_optimized, W_D_optimized,
                       tf.complex(csi_rx[:, :, :, :, :, :, 0], csi_rx[:, :, :, :, :, :, 1]),
                       V_RF_optimized, W_RF_optimized, y_symbols_k0]
            y_symbols_k0_equalized, noise_effective_k0 = self.equalizer(inputs3)
            noise_effective_k0_reshaped = tf.expand_dims(noise_effective_k0, axis=5)
            llr_k0 = self.demapper(
                [y_symbols_k0_equalized / self.rescaling, noise_effective_k0_reshaped])  # self.sigma2 / (2 * np.pi)

            # BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, M 2, K_prime 3, Nue 4, N_s 5, log2(M) 6
            inputs1 = [
                tx_bits[batch_number * self.setup.BATCHSIZE: (batch_number + 1) * self.setup.BATCHSIZE, :, :, 0, :, :,
                :],
                llr_k0]  # BATCHSIZE 0, int(Nsymb * sampling_ratio_time_domain_keep) 1, M 2, Nue 3, N_s 4, log2(M) 5

            bmi_avg, bmi = self.bmi_in_presence_of_phase_noise(inputs1)

            if (batch_number == 0):
                air_samples_k0 = bmi
            else:
                air_samples_k0 = tf.concat([air_samples_k0, bmi], axis=0)

        return air_samples_k0, y_symbols_k0_equalized / self.rescaling

    @property
    def metrics(self):
        return [loss_metric, loss_metric_test, capacity_metric_test]
