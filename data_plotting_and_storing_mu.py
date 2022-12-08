import numpy as np
import tensorflow as tf
from sionna.mapping import Constellation, Mapper, Demapper
import sionna


class Data_plotting_and_storing():
    def __init__(self, obj_ML_model_post_training, constellation, tx_symbols, tx_bits, setup):
        super(Data_plotting_and_storing, self).__init__()
        self.setup = setup
        self.obj_ML_model_post_training = obj_ML_model_post_training
        self.constellation = constellation
        self.tx_symbols = tx_symbols
        self.tx_bits = tx_bits
        tf.config.run_functions_eagerly(True)
        self.demapper = Demapper("app", constellation=self.constellation, hard_out=True)
        self.ber_fcn = sionna.utils.BitErrorRate(name='bit_error_rate')

    @tf.function
    def execute_for_Sohrabi(self):
        ber_fcn = sionna.utils.BitErrorRate(name='bit_error_rate')
        air_samples, rx_symbols = self.obj_ML_model_post_training.evaluation_of_Sohrabis_beamformer(self.tx_bits, self.tx_symbols)

        BER = []
        MSE = []
        for ns in range(round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep)):
            BER_u = []
            MSE_u = []
            for u in range(self.setup.Nue):
                flattened_rx_symbols = tf.reshape(rx_symbols[:, ns, u, :, :, :], [-1])
                flattened_received_bits = self.demapper([flattened_rx_symbols, self.setup.sigma2 / (2 * np.pi)])
                flattened_tx_symbols = tf.reshape(self.tx_symbols[:, ns, :, 0, u, :, :], [-1])
                flattened_transmitted_bits = tf.reshape(self.tx_bits[:, ns, :, 0, u, :, :], [-1])
                BER_u.append(ber_fcn(flattened_received_bits, flattened_transmitted_bits))
                MSE_u.append(tf.reduce_mean(tf.square(tf.abs(flattened_rx_symbols - flattened_tx_symbols))))

            BER.append(tf.stack(BER_u, axis=0))
            MSE.append(tf.stack(MSE_u, axis=0))

        return air_samples, rx_symbols, tf.stack(BER, axis=0), tf.stack(MSE, axis=0)


    @tf.function
    def execute_for_DBF(self):
        ber_fcn = sionna.utils.BitErrorRate(name='bit_error_rate')
        air_samples, rx_symbols = self.obj_ML_model_post_training.evaluation_of_digital_beamformer(
            self.tx_bits, self.tx_symbols)
        BER = []
        MSE = []
        for ns in range(round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep)):
            BER_u = []
            MSE_u = []
            for u in range(self.setup.Nue):
                flattened_rx_symbols = tf.reshape(rx_symbols[:, ns, u, :, :, :], [-1])
                flattened_received_bits = self.demapper([flattened_rx_symbols, self.setup.sigma2 / (2 * np.pi)])
                flattened_tx_symbols = tf.reshape(self.tx_symbols[:, ns, :, 0, u, :, :], [-1])
                flattened_transmitted_bits = tf.reshape(self.tx_bits[:, ns, :, 0, u, :, :], [-1])
                BER_u.append(ber_fcn(flattened_received_bits, flattened_transmitted_bits))
                MSE_u.append(tf.reduce_mean(tf.square(tf.abs(flattened_rx_symbols - flattened_tx_symbols))))

            BER.append(tf.stack(BER_u, axis=0))
            MSE.append(tf.stack(MSE_u, axis=0))

        return air_samples, rx_symbols, tf.stack(BER, axis=0), tf.stack(MSE, axis=0)

    @tf.function
    def execute_for_proposed(self):
        ber_fcn = sionna.utils.BitErrorRate(name='bit_error_rate')
        air_samples, rx_symbols = self.obj_ML_model_post_training.evaluation_of_proposed_beamformer(
            self.tx_bits, self.tx_symbols)

        BER = []
        MSE = []
        for ns in range(round(self.setup.CSIRSPeriod * self.setup.sampling_ratio_time_domain_keep)):
            BER_u = []
            MSE_u = []
            for u in range(self.setup.Nue):
                flattened_rx_symbols = tf.reshape(rx_symbols[:, ns, u, :, :, :], [-1])
                flattened_received_bits = self.demapper([flattened_rx_symbols, self.setup.sigma2 / (2 * np.pi)])
                flattened_tx_symbols = tf.reshape(self.tx_symbols[:, ns, :, 0, u, :, :], [-1])
                flattened_transmitted_bits = tf.reshape(self.tx_bits[:, ns, :, 0, u, :, :], [-1])
                BER_u.append(ber_fcn(flattened_received_bits, flattened_transmitted_bits))
                MSE_u.append(tf.reduce_mean(tf.square(tf.abs(flattened_rx_symbols - flattened_tx_symbols))))

            BER.append(tf.stack(BER_u, axis=0))
            MSE.append(tf.stack(MSE_u, axis=0))

        return air_samples, rx_symbols, tf.stack(BER, axis=0), tf.stack(MSE, axis=0)
