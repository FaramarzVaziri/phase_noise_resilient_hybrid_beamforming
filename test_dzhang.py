# External libs
import scipy.io as sio
import tensorflow as tf
import numpy as np
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, hard_decisions

try:  # Using TPU if available
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    # strategy = tf.distribute.experimental.TPUStrategy(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
except ValueError:  # Using CPU if TPU is not available, also, using CPU, the run is faster in eager mode
    strategy = tf.distribute.get_strategy()
    tf.config.run_functions_eagerly(True)

# internal libs
from common_setup_mu import CommonSetUp
from dnn_mu import CNN_model_class
from ml_model_mu import ML_model_class
from dataset_generator_mu import DatasetGenerator
from rx_ops_and_losses_mu import RX_ops_and_losses
from data_plotting_and_storing_mu import Data_plotting_and_storing
from the_clock import Timer_class
from loss_phase_noise_free import Loss_phase_noise_free

if __name__ == '__main__':
    # Control panel 1 (danger zone!) ----------------------------------
    use_test_data_for_train = False
    llr_learnable = 'no'  # feature not implemented
    choise_of_loss = "bmi"
    rx_refresher = 'no'  # feature not implemented
    equalizer_type = 'LMMSE'
    use_attention = 'yes'
    truncation_ratio_keep = 1  # bypassed
    sampling_ratio_subcarrier_domain_keep = 1  # bypassed
    GMI_approx = 'no' # feature not implemented
    K_prime_size_test = 'no' # bypassed



    # Control panel 2 (for normal use) ----------------------------------
    # training parameters
    load_trained_best_model = 'no'
    do_train = 'no'
    save_model = 'no'
    evaluate_model = 'no'
    n_epochs = 5
    validation_freq = 1

    # Benchmark methods
    benchmark = "Dzhang"
    gather_data_for_running_benchmark = 'no'
    evaluate_benchmark = 'yes'
    evaluate_DBF = 'no'

    # dateset
    is_data_segmented = 'no'
    create_DS_phase_noised = 'yes'
    train_dataset_size = 4
    train_data_fragment_size = train_dataset_size
    test_dataset_size = 4
    test_data_fragment_size = test_dataset_size
    eval_dataset_size = 128
    eval_data_fragment_size = eval_dataset_size

    # optimization parameters
    L_rate_initial = 6e-5
    BATCHSIZE = 128
    gradient_norm_clipper_pre = 1.
    gradient_norm_clipper_post = 1.
    ReduceLROnPlateau_decay_rate = 0.5
    ReduceLROnPlateau_patience = 1
    ReduceLROnPlateau_min_lr = 1e-7

    # neural net
    convolutional_kernels = 3
    extra_kernel = 0  # for input and output layers
    convolutional_filters = 64
    convolutional_strides = 1
    convolutional_dilation = 1
    subcarrier_strides_l1 = 1
    N_u_a_strides_l1 = 1
    N_b_a_strides_l1 = 1
    subcarrier_strides_l2 = 1
    N_u_a_strides_l2 = 1
    N_b_a_strides_l2 = 1
    n_common_layers = 10
    n_D_and_RF_layers = 10
    n_post_Tconv_processing = 2
    n_llr_DNN_learner =2  # not used

    # MIMO-OFDM
    mod_type = '16QAM'
    if mod_type == 'QPSK':
        M = 4
    elif mod_type == '16QAM':
        M = 16
    elif mod_type == '64QAM':
        M = 64

    N_s = 1
    Nue = 4
    N_b_a = 16
    N_u_a = 4
    N_b_rf = 4
    N_u_rf = 1
    N_b_o = N_b_rf
    N_u_o = N_u_rf
    K = 1024
    K_prime = 4  # size of the influencial subcarriers set
    PTRS_seperation = round(
        K_prime / 2)  # Separation between PTRS subcarriers. I abandoned the idea of refreshing RX CSI using PTRS (so it is not used)
    E_tx_dBm_per_Hz = -60 # Power budget per subcarrier
    P = 10**((E_tx_dBm_per_Hz - 30.) / 10.)
    sigma2_dBm_per_Hz = -139
    sigma2 = 10**((sigma2_dBm_per_Hz - 30) / 10)
    apply_channel_est_error = False
    channel_est_err_mse_per_element_dB = -100.

    # Phase noise parameters
    CSIRSPeriod = 20*14 # 20 subframes, 14 symbols per subframe
    f_0 = 100e3
    L = -100
    fs = 10 ** 9 / 65.104
    phase_noise_recorded = 'no'
    CLO_or_ILO = 'ILO'

    # simulation parameters

    sampling_ratio_time_domain_keep = 5 / CSIRSPeriod
    sampling_ratio_time_domain_keep_capacity_metric = sampling_ratio_time_domain_keep
    influencial_subcarriers_set_size = 4

    # end of control panel ----------------------------------




    setup = CommonSetUp(use_test_data_for_train
                        , load_trained_best_model
                        , llr_learnable
                        , choise_of_loss
                        , rx_refresher
                        , equalizer_type
                        , use_attention
                        , do_train
                        , save_model
                        , evaluate_model
                        , n_epochs
                        , validation_freq
                        , benchmark
                        , gather_data_for_running_benchmark
                        , evaluate_benchmark
                        , evaluate_DBF
                        , create_DS_phase_noised
                        , train_dataset_size
                        , train_data_fragment_size
                        , test_dataset_size
                        , test_data_fragment_size
                        , eval_dataset_size
                        , eval_data_fragment_size
                        , is_data_segmented
                        , L_rate_initial
                        , BATCHSIZE
                        , gradient_norm_clipper_pre
                        , gradient_norm_clipper_post
                        , ReduceLROnPlateau_decay_rate
                        , ReduceLROnPlateau_patience
                        , ReduceLROnPlateau_min_lr
                        , convolutional_kernels
                        , extra_kernel
                        , convolutional_filters
                        , convolutional_strides
                        , convolutional_dilation
                        , subcarrier_strides_l1
                        , N_u_a_strides_l1
                        , N_b_a_strides_l1
                        , subcarrier_strides_l2
                        , N_u_a_strides_l2
                        , N_b_a_strides_l2
                        , n_common_layers
                        , n_D_and_RF_layers
                        , n_post_Tconv_processing
                        , n_llr_DNN_learner
                        , mod_type
                        , M
                        , N_b_a
                        , N_b_rf
                        , N_b_o
                        , N_u_a
                        , N_u_rf
                        , N_u_o
                        , N_s
                        , Nue
                        , K
                        , K_prime
                        , PTRS_seperation
                        , P
                        , sigma2
                        , apply_channel_est_error
                        , channel_est_err_mse_per_element_dB
                        , CSIRSPeriod
                        , f_0
                        , L
                        , fs
                        , phase_noise_recorded
                        , CLO_or_ILO
                        , truncation_ratio_keep
                        , sampling_ratio_time_domain_keep
                        , sampling_ratio_time_domain_keep_capacity_metric
                        , sampling_ratio_subcarrier_domain_keep
                        , GMI_approx
                        , K_prime_size_test
                        , influencial_subcarriers_set_size)

    constellation = Constellation("qam", num_bits_per_symbol=round(np.log2(setup.M)), normalize=True, center=False)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper("app", constellation=constellation, hard_out=False)
    # Binary source that generates random 0s/1s
    source = BinarySource()
    tx_bits = tf.constant([[0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 1, 1],
                           [0, 1, 0, 0],
                           [0, 1, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 1, 1],
                           [1, 0, 0, 0],
                           [1, 0, 0, 1],
                           [1, 0, 1, 0],
                           [1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0],
                           [1, 1, 1, 1]], dtype=tf.float32)
    tx_bits = tf.reshape(tx_bits, shape=[1, 1, setup.M, 1, 1, 1, int(np.log2(setup.M))])
    tx_bits_train = tf.tile(tx_bits,
                            multiples=[setup.BATCHSIZE,
                                       int(setup.CSIRSPeriod * setup.sampling_ratio_time_domain_keep),
                                       1,
                                       setup.K_prime,
                                       setup.Nue,
                                       setup.N_s,
                                       1])
    tx_symbols_train = mapper(tx_bits_train)
    tx_bits_eval = tf.tile(tx_bits,
                           multiples=[setup.eval_dataset_size,
                                      int(setup.CSIRSPeriod * setup.sampling_ratio_time_domain_keep),
                                      1,
                                      setup.K_prime,
                                      setup.Nue,
                                      setup.N_s,
                                      1])
    tx_symbols_eval = mapper(tx_bits_eval)

    for train_id in range(0, 1, 1):  # in case of sharding this loop sweeps the different shards

        dataset_id_start = round(setup.train_dataset_size_post / setup.train_data_fragment_size_post) * train_id

        test_DS_gen = DatasetGenerator(train_or_test='test',
                                       phase_noise_exists='yes',
                                       data_fragment_size=setup.test_data_fragment_size,
                                       dataset_id_start=0,
                                       evaluation_bits=tx_bits_eval,
                                       evaluation_symbols=tx_symbols_eval
                                       , setup=setup)
        dataset_test = test_DS_gen.dataset_generator()
        cnn = CNN_model_class(setup=setup)
        the_model = cnn.CNN_transceiver_AdaSE_ResNet_with_binding(trainable_csi=True, layer_name='TRX')

        loss_phn_free = Loss_phase_noise_free(setup=setup)
        loss_function_phn_free = loss_phn_free.capacity_forall_samples

        rx_ops_and_losses_obj = RX_ops_and_losses(constellation=constellation, demapper=demapper, setup=setup)

        capacity_metric = rx_ops_and_losses_obj.BMI_metric
        capacity_metric_2 = rx_ops_and_losses_obj.BMI_metric
        equalizer = rx_ops_and_losses_obj.equalizer_forall_samples

        # pre training is not used anymore. I only use it to load the weights of the pre-trained model to the new model
        ml_model_pre_training = ML_model_class(CNN_transceiver=the_model,
                                               obj_test_dataset=test_DS_gen,
                                               tx_bits_train=tx_bits_train,
                                               tx_symbols_train=tx_symbols_train,
                                               demapper=demapper,
                                               equalizer=equalizer,
                                               setup=setup)

        optimizer_pre = tf.keras.optimizers.Adam(learning_rate=setup.L_rate_initial,
                                                 clipnorm=setup.gradient_norm_clipper_pre)

        tf.keras.utils.plot_model(the_model, show_shapes=True, show_layer_names=True, to_file='cnn_trx.png')
        the_model.summary()
        n_params = the_model.count_params()
        n_layers = len(the_model.layers)

        ml_model_pre_training.compile(
            optimizer=optimizer_pre,
            loss=loss_function_phn_free,
            rx_calc=[],
            activation_TX=cnn.custom_activation_transmitter,
            activation_RX=cnn.custom_activation_receiver,
            bmi_in_presence_of_phase_noise=capacity_metric,
            capacity_in_presence_of_phase_noise=capacity_metric_2)

        reduce_lr_pre = tf.keras.callbacks.ReduceLROnPlateau(monitor='L2_train',
                                                             factor=setup.ReduceLROnPlateau_decay_rate,
                                                             patience=setup.ReduceLROnPlateau_patience,
                                                             min_lr=setup.ReduceLROnPlateau_min_lr,
                                                             mode='min', verbose=1)

        if setup.load_trained_best_model == 'yes':
            ml_model_pre_training.built = True
            ml_model_pre_training.load_weights(setup.address_of_best_model)

        if setup.create_DS_phase_noised == 'yes':
            train_DS_gen_phned = DatasetGenerator(train_or_test='train',
                                                  phase_noise_exists='yes',
                                                  data_fragment_size=setup.train_data_fragment_size_post,
                                                  dataset_id_start=dataset_id_start,
                                                  evaluation_bits=[],
                                                  evaluation_symbols=[]
                                                  , setup=setup)
            dataset_train_phned = train_DS_gen_phned.dataset_generator()

        if setup.choise_of_loss == "Frobinous":
            loss_function_phned = rx_ops_and_losses_obj.Frobinious_distance_of_modulated_signals
        elif setup.choise_of_loss == "bmi":
            loss_function_phned = rx_ops_and_losses_obj.BMI_loss
        else:
            print('-- WRONG LOSS FUNCTION')

        # Transfer learning
        ml_model_post_training = ML_model_class(CNN_transceiver=the_model,
                                                obj_test_dataset=test_DS_gen,
                                                tx_bits_train=tx_bits_train,
                                                tx_symbols_train=tx_symbols_train,
                                                demapper=demapper,
                                                equalizer=equalizer
                                                , setup=setup)

        optimizer_post = tf.keras.optimizers.Adam(learning_rate=setup.L_rate_initial,
                                                  clipnorm=setup.gradient_norm_clipper_post)
        ml_model_post_training.compile(
            optimizer=optimizer_post,
            loss=loss_function_phned,
            rx_calc=rx_ops_and_losses_obj.rx_calc,
            activation_TX=cnn.custom_activation_transmitter,
            activation_RX=cnn.custom_activation_receiver,
            bmi_in_presence_of_phase_noise=capacity_metric,
            capacity_in_presence_of_phase_noise=capacity_metric_2)

        if setup.load_trained_best_model == 'yes':
            ml_model_post_training.built = True
            ml_model_post_training.load_weights(setup.address_of_best_model)

        reduce_lrTL = tf.keras.callbacks.ReduceLROnPlateau(monitor='L2_train',
                                                           factor=setup.ReduceLROnPlateau_decay_rate,
                                                           patience=setup.ReduceLROnPlateau_patience,
                                                           min_lr=setup.ReduceLROnPlateau_min_lr,
                                                           mode='min', verbose=1)

        if setup.do_train == 'yes':
            if setup.use_test_data_for_train == False:
                h_post = ml_model_post_training.fit(dataset_train_phned,
                                                    epochs=setup.epochs_post,
                                                    callbacks=[reduce_lrTL],
                                                    validation_data=dataset_test,
                                                    validation_freq=setup.val_freq_post,
                                                    verbose=1)
            elif setup.use_test_data_for_train == True:
                h_post = ml_model_post_training.fit(dataset_test,
                                                    epochs=setup.epochs_post,
                                                    callbacks=[reduce_lrTL],
                                                    validation_data=dataset_test,
                                                    validation_freq=setup.val_freq_post,
                                                    verbose=1)  #


            if setup.save_post == 'yes':
                ml_model_post_training.save_weights(setup.address_of_best_model)
                print("-- trained model is saved.")

    if setup.gather_data_for_running_benchmark == 'yes':
        HH_complex, ccsi_tx, HH_tilde, LLambda_B, LLambda_U =\
            test_DS_gen.data_generator_for_running_benchmark_beamformer()
        H_complex = HH_complex.numpy()
        csi_tx = ccsi_tx.numpy()
        H_tilde = HH_tilde.numpy()
        Lambda_B = LLambda_B.numpy()
        Lambda_U = LLambda_U.numpy()
        mdic_data_for_sunning_soh = {"H": H_complex,
                                     "csi_tx": csi_tx,
                                     "H_tilde": H_tilde,
                                     'Lambda_B': Lambda_B,
                                     'Lambda_U': Lambda_U}

        sio.savemat(setup.dataset_for_running_benchmark, mdic_data_for_sunning_soh)
        print('-- data gatherd for benchmark methods')

    obj_data_plotting_and_storing = Data_plotting_and_storing(obj_ML_model_post_training=ml_model_post_training,
                                                              constellation=constellation,
                                                              tx_symbols=tx_symbols_eval,
                                                              tx_bits=tx_bits_eval,
                                                              setup=setup)


    if setup.evaluate_benchmark == 'yes':
        air_samples_Soh, rx_symbols_Soh, BER_Soh, MSE_Soh = obj_data_plotting_and_storing.execute_for_benchmark()
        mdic_soh = {"C_samples_x_OFDM_index": air_samples_Soh.numpy(),
                    'L': 0,
                    'tx_symbols': tx_symbols_eval.numpy(),
                    'rx_symbols': rx_symbols_Soh.numpy(),
                    'BER': BER_Soh.numpy(),
                    'MSE': MSE_Soh.numpy()}
        sio.savemat(setup.eval_file_name_benchmark, mdic_soh)
        print('benchmark beamformer is evaluated')

    if setup.evaluate_DBF == 'yes':
        air_samples_DBF, rx_symbols_DBF, BER_DBF, MSE_DBF = obj_data_plotting_and_storing.execute_for_DBF()
        mdic_DBF = {"C_samples_x_OFDM_index": air_samples_DBF.numpy(),
                    'L': 0,
                    'tx_symbols': tx_symbols_eval.numpy(),
                    'rx_symbols': rx_symbols_DBF.numpy(),
                    'BER': BER_DBF.numpy(),
                    'MSE': MSE_DBF.numpy()}
        sio.savemat(setup.eval_file_name_DBF, mdic_DBF)
        print('DBF beamformer is evaluated')


    if setup.evaluate_post == 'yes':
        air_samples_proposed, rx_symbols_proposed, BER_proposed, MSE_proposed = obj_data_plotting_and_storing.execute_for_proposed()
        mdic_proposed = {"C_samples_x_OFDM_index": air_samples_proposed.numpy(),
                         'L': 0,
                         'tx_symbols': tx_symbols_eval.numpy(),
                         'rx_symbols': rx_symbols_proposed.numpy(),
                         'BER': BER_proposed.numpy(),
                         'MSE': MSE_proposed.numpy()}
        sio.savemat(setup.eval_file_name, mdic_proposed)
        print('proposed beamformer is evaluated')

