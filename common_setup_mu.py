import math


class CommonSetUp():
    def __init__(self,
                 use_test_data_for_train
                 , load_trained_best_model
                 , llr_learnable
                 , choise_of_loss
                 , rx_refresher
                 , equalizer_type
                 , use_attention
                 , do_train
                 , save_post
                 , evaluate_post
                 , epochs_post
                 , val_freq_post
                 , benchmark
                 , gather_data_for_running_benchmark
                 , evaluate_benchmark
                 , evaluate_DBF
                 , create_DS_phase_noised
                 , train_dataset_size_post
                 , train_data_fragment_size_post
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
                 , SNR
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
                 , influencial_subcarriers_set_size):
        super(CommonSetUp, self).__init__()

        self.use_test_data_for_train = use_test_data_for_train
        self.load_trained_best_model = load_trained_best_model
        self.llr_learnable = llr_learnable
        self.choise_of_loss = choise_of_loss
        self.rx_refresher = rx_refresher
        self.equalizer_type = equalizer_type
        self.use_attention = use_attention

        self.do_train = do_train
        self.save_post = save_post
        self.evaluate_post = evaluate_post
        self.epochs_post = epochs_post
        self.val_freq_post = val_freq_post

        self.benchmark = benchmark
        self.gather_data_for_running_benchmark = gather_data_for_running_benchmark
        self.evaluate_benchmark = evaluate_benchmark
        self.evaluate_DBF = evaluate_DBF

        self.create_DS_phase_noised = create_DS_phase_noised
        self.train_dataset_size_post = train_dataset_size_post
        self.train_data_fragment_size_post = train_data_fragment_size_post
        self.test_dataset_size = test_dataset_size
        self.test_data_fragment_size = test_data_fragment_size
        self.eval_dataset_size = eval_dataset_size
        self.eval_data_fragment_size = eval_data_fragment_size
        self.is_data_segmented = is_data_segmented

        self.L_rate_initial = L_rate_initial
        self.BATCHSIZE = BATCHSIZE
        self.gradient_norm_clipper_pre = gradient_norm_clipper_pre
        self.gradient_norm_clipper_post = gradient_norm_clipper_post
        self.ReduceLROnPlateau_decay_rate = ReduceLROnPlateau_decay_rate
        self.ReduceLROnPlateau_patience = ReduceLROnPlateau_patience
        self.ReduceLROnPlateau_min_lr = ReduceLROnPlateau_min_lr

        self.convolutional_kernels = convolutional_kernels
        self.extra_kernel = extra_kernel
        self.convolutional_filters = convolutional_filters
        self.convolutional_strides = convolutional_strides
        self.convolutional_dilation = convolutional_dilation
        self.subcarrier_strides_l1 = subcarrier_strides_l1
        self.N_u_a_strides_l1 = N_u_a_strides_l1
        self.N_b_a_strides_l1 = N_b_a_strides_l1
        self.subcarrier_strides_l2 = subcarrier_strides_l2
        self.N_u_a_strides_l2 = N_u_a_strides_l2
        self.N_b_a_strides_l2 = N_b_a_strides_l2
        self.n_common_layers = n_common_layers
        self.n_D_and_RF_layers = n_D_and_RF_layers
        self.n_post_Tconv_processing = n_post_Tconv_processing
        self.n_llr_DNN_learner = n_llr_DNN_learner

        self.mod_type = mod_type
        self.M = M
        self.N_b_a = N_b_a
        self.N_b_rf = N_b_rf
        self.N_b_o = N_b_o
        self.N_u_a = N_u_a
        self.N_u_rf = N_u_rf
        self.N_u_o = N_u_o
        self.N_s = N_s
        self.Nue = Nue
        self.K = K
        self.K_prime = K_prime
        self.PTRS_seperation = PTRS_seperation
        self.SNR = SNR
        self.P = P
        self.sigma2 = sigma2
        self.apply_channel_est_error = apply_channel_est_error
        self.channel_est_err_mse_per_element_dB = channel_est_err_mse_per_element_dB
        self.CSIRSPeriod = CSIRSPeriod
        self.f_0 = f_0
        self.L = L
        self.fs = fs
        self.Ts = 1. / self.fs
        self.phase_noise_power_augmentation_factor = 1.
        self.PHN_innovation_std = self.phase_noise_power_augmentation_factor \
                                  * math.sqrt(4.0 * math.pi ** 2 * self.f_0 ** 2 * 10 ** (self.L / 10.) * self.Ts)
        self.phase_noise_recorded = phase_noise_recorded
        self.CLO_or_ILO = CLO_or_ILO

        # file IDs
        self.simulation_ID = "N_ue_" + str(self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(
            self.N_u_a) + "_N_b_rf_" + str(self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_M_" + str(self.M) + "_phase_noise_recorded_" + str(
            self.phase_noise_recorded) + "_" + str(self.CLO_or_ILO) + "_ch_est_err_var_dB_" + str(int(self.channel_est_err_mse_per_element_dB))

        self.model_ID = "N_ue_" + str(self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(
            self.N_u_a) + "_N_b_rf_" + str(self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_M_" + str(self.M) + "_phase_noise_recorded_" + str(
            self.phase_noise_recorded) + "_" + str(self.CLO_or_ILO)

        # dateset
        self.dataset_name = "datasets/DS" + "_N_ue_" + str(self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(
            self.N_u_a) + "_K_prime_" + str(self.K_prime) + ".mat"
        self.dataset_phi_name = "PN_data_73GHz_resampled"
        self.dataset_phi_address = "datasets/" + self.dataset_phi_name + ".mat"

        self.dataset_for_running_benchmark = "datasets/DS_for_running_benchmark_" + "N_ue_" + str(
            self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(self.N_u_a) + "_N_b_rf_" + str(
            self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_ch_est_err_var_dB_" + str(int(self.channel_est_err_mse_per_element_dB)) + ".mat"
        self.dataset_for_testing_sohrabi = "datasets/DS_for_testing_Sohrabi_" + "N_ue_" + str(
            self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(self.N_u_a) + "_N_b_rf_" + str(
            self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_ch_est_err_var_dB_" + str(int(self.channel_est_err_mse_per_element_dB)) + ".mat"
        self.dataset_for_testing_DBF = self.dataset_for_testing_sohrabi
        self.dataset_for_testing_zilli = "datasets/DS_for_testing_Zilli_" + "N_ue_" + str(
            self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(self.N_u_a) + "_N_b_rf_" + str(
            self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_ch_est_err_var_dB_" + str(
            int(self.channel_est_err_mse_per_element_dB)) + ".mat"
        self.dataset_for_testing_Dzhang = "datasets/DS_for_testing_Dzhang_" + "N_ue_" + str(
            self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(self.N_u_a) + "_N_b_rf_" + str(
            self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_ch_est_err_var_dB_" + str(
            int(self.channel_est_err_mse_per_element_dB)) + ".mat"
        self.dataset_for_testing_Gonzales = "datasets/DS_for_testing_Gonzales_" + "N_ue_" + str(
            self.Nue) + "_N_b_a_" + str(self.N_b_a) + "_N_u_a_" + str(self.N_u_a) + "_N_b_rf_" + str(
            self.N_b_rf) + "_N_u_rf_" + str(self.N_u_rf) + "_N_s_" + str(
            self.N_s) + "_L_" + str(int(self.L)) + "_SNR_" + str(int(self.SNR)) + "_K_" + str(
            self.K) + "_K_prime_" + str(self.K_prime) + "_ch_est_err_var_dB_" + str(
            int(self.channel_est_err_mse_per_element_dB)) + ".mat"

        if self.benchmark == "Sohrabi":
            self.dataset_for_testing_benchmark = self.dataset_for_testing_sohrabi
        elif self.benchmark == "Zilli":
            self.dataset_for_testing_benchmark = self.dataset_for_testing_zilli
        elif self.benchmark == "Dzhang":
            self.dataset_for_testing_benchmark = self.dataset_for_testing_Dzhang
        elif self.benchmark == "Gonzales":
            self.dataset_for_testing_benchmark = self.dataset_for_testing_Gonzales

        # simulation results
        self.eval_file_name = "results/eval_" + self.simulation_ID + ".mat"
        self.eval_file_name_benchmark = "results/eval_" + self.benchmark + "_" + self.simulation_ID + ".mat"
        self.eval_file_name_DBF = "results/eval_DBF_" + self.simulation_ID + ".mat"

        self.address_of_best_model = "models/SE_ResNet_" + self.model_ID + ".h5"

        self.truncation_ratio_keep = truncation_ratio_keep
        self.sampling_ratio_time_domain_keep = sampling_ratio_time_domain_keep
        self.sampling_ratio_time_domain_keep_capacity_metric = sampling_ratio_time_domain_keep_capacity_metric
        self.sampling_ratio_subcarrier_domain_keep = sampling_ratio_subcarrier_domain_keep
        self.GMI_approx = GMI_approx

        self.K_prime_size_test = K_prime_size_test
        self.influencial_subcarriers_set_size = influencial_subcarriers_set_size
