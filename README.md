# Instructions for using the code in this repository

## Recommended packages:
['absl-py==0.13.0', 'aiohttp==3.7.4', 'anyio==3.5.0', 'argon2-cffi-bindings==21.2.0', 'argon2-cffi==21.3.0', 'astor==0.8.1', 'asttokens==2.0.5', 'astunparse==1.6.3', 'async-timeout==3.0.1', 'attrs==21.2.0', 'autopep8==1.6.0', 'babel==2.9.1', 'backcall==0.2.0', 'beautifulsoup4==4.11.1', 'bleach==4.1.0', 'blinker==1.4', 'brotlipy==0.7.0', 'cached-property==1.5.2', 'cachetools==4.2.2', 'certifi==2022.6.15', 'cffi==1.14.6', 'chardet==3.0.4', 'charset-normalizer==2.0.7', 'click==8.0.4', 'colorama==0.4.5', 'coverage==5.5', 'cryptography==3.4.7', 'cycler==0.10.0', 'cython==0.29.24', 'debugpy==1.5.1', 'decorator==5.1.1', 'defusedxml==0.7.1', 'dm-tree==0.1.6', 'entrypoints==0.4', 'et-xmlfile==1.1.0', 'executing==0.8.3', 'fastjsonschema==2.15.1', 'flask==2.0.2', 'flatbuffers==20210226132247', 'gast==0.4.0', 'google-auth-oauthlib==0.4.1', 'google-auth==1.33.0', 'google-pasta==0.2.0', 'grpcio==1.42.0', 'h5py==3.2.1', 'idna==3.3', 'importlib-metadata==3.10.0', 'importlib-resources==5.4.0', 'ipykernel==6.9.1', 'ipython-genutils==0.2.0', 'ipython==8.4.0', 'ipywidgets==7.6.5', 'itsdangerous==2.0.1', 'jax==0.2.18', 'jedi==0.18.1', 'jinja2==3.0.3', 'json5==0.9.6', 'jsonschema==4.4.0', 'jupyter-client==7.2.2', 'jupyter-console==6.4.3', 'jupyter-core==4.10.0', 'jupyter-server==1.18.1', 'jupyter==1.0.0', 'jupyterlab-pygments==0.1.2', 'jupyterlab-server==2.12.0', 'jupyterlab-widgets==1.0.0', 'jupyterlab==3.4.4', 'keras-preprocessing==1.1.2', 'keras==2.4.3', 'kiwisolver==1.3.1', 'line-profiler==3.3.0', 'llvmlite==0.36.0', 'markdown==3.3.4', 'markupsafe==2.1.1', 'matplotlib-inline==0.1.2', 'matplotlib==3.4.2', 'mistune==0.8.4', 'mkl-fft==1.3.0', 'mkl-random==1.2.2', 'mkl-service==2.4.0', 'mpmath==1.2.1', 'multidict==5.1.0', 'nbclassic==0.3.5', 'nbclient==0.5.13', 'nbconvert==6.4.4', 'nbformat==5.3.0', 'nest-asyncio==1.5.5', 'notebook==6.4.12', 'numba==0.53.1', 'numpy==1.21.3', 'oauthlib==3.1.1', 'olefile==0.46', 'openai==0.19.0', 'openpyxl==3.0.9', 'opt-einsum==3.3.0', 'packaging==21.3', 'pandas-stubs==1.2.0.35', 'pandas==1.3.4', 'pandocfilters==1.5.0', 'parso==0.8.3', 'pickleshare==0.7.5', 'pillow==8.3.1', 'pip==21.1.3', 'prometheus-client==0.14.1', 'prompt-toolkit==3.0.20', 'protobuf==3.14.0', 'pure-eval==0.2.2', 'pyasn1-modules==0.2.8', 'pyasn1==0.4.8', 'pycodestyle==2.8.0', 'pycparser==2.20', 'pydot==1.4.1', 'pygments==2.11.2', 'pyjwt==2.1.0', 'pyopenssl==20.0.1', 'pyparsing==2.4.7', 'pyqt5-sip==4.19.18', 'pyqt5==5.12.3', 'pyqtchart==5.12', 'pyqtwebengine==5.12.1', 'pyreadline==2.1', 'pyrsistent==0.18.0', 'pysocks==1.7.1', 'python-dateutil==2.8.2', 'python-dotenv==0.19.2', 'pytz==2022.1', 'pywin32==302', 'pywinpty==2.0.2', 'pyyaml==5.4.1', 'pyzmq==23.2.0', 'qtconsole==5.3.1', 'qtpy==2.0.1', 'requests-oauthlib==1.3.0', 'requests==2.28.1', 'rsa==4.7.2', 'scipy==1.6.2', 'send2trash==1.8.0', 'setuptools==52.0.0.post20210125', 'sionna==0.8.0', 'six==1.16.0', 'sniffio==1.2.0', 'soupsieve==2.3.1', 'stack-data==0.2.0', 'sympy==1.9', 'tensorboard-data-server==0.6.1', 'tensorboard-plugin-wit==1.6.0', 'tensorboard==2.5.0', 'tensorflow-addons==0.14.0', 'tensorflow-estimator==2.5.0', 'tensorflow-model-optimization==0.7.0', 'tensorflow==2.5.0', 'termcolor==1.1.0', 'terminado==0.13.1', 'testpath==0.6.0', 'toml==0.10.2', 'tornado==6.1', 'tqdm==4.62.3', 'traitlets==5.1.1', 'typeguard==2.12.1', 'typing-extensions==3.10.0.0', 'urllib3==1.26.11', 'wcwidth==0.2.5', 'webencodings==0.5.1', 'websocket-client==0.58.0', 'werkzeug==2.0.3', 'wheel==0.35.1', 'widgetsnbextension==3.5.2', 'win-inet-pton==1.1.0', 'wincertstore==0.2', 'wrapt==1.12.1', 'yarl==1.6.3', 'zipp==3.5.0']

# Control panel 1 (danger zone!)
The following controlling capabilities are not for normal use. These are there for some experimental features and sanity checks.
They are not documented here. If you want to use them, you have to read the source code.

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

# Control panel 2 (for normal use)

The following controlling capabilities are for normal use.
## Training parameters
- load_trained_best_model: Before training **_load_trained_best_model_** should be set to 'no' because there is no existing model to load. 
After training, if **_evaluate_model_** is 'yes', **_load_trained_best_model_** should
be set to 'yes' to evaluate the trained model. 
To maintain the reproducibility of the training sessions, a good practice is to avoid transfer learning, i.e., 
avoid using a trained model as an initial point for a new training session. If you always leave
**_load_trained_best_model_** to 'yes', you will always use the latest trained model as the initial
point for your next training. However, it is a handy tool for breaking down lengthy training sessions into multiple 
smaller ones to apply manual hyper-parameter tuning during the training.
- do_train, save_model, evaluate_model: These are the main knobs to control the training, saving and evaluation of the
model. Notice that each model is saved after going through all epochs. Interrupting the training will result in losing 
the model parameters until the last saved model. A good practice is to run the training in graph mode and evaluation in eager mode. This achieves the final results
the fastest. The code is written in such a way that it switches between graph and eager modes based on
**_do_train_**. If it is 'yes', it runs in graph mode. If it is 'no', it runs in eager mode. So, it is best to avoid
training and evaluating the model at the same time.
- n_epochs, validation_freq: **_n_epochs_** is the number of epochs to train the model and **_validation_freq_**
 specifies the periodicity of validation in number of epochs for the purpose of adaptive learning rate adjustment. 
For a dataset of size 1024 (which gets augmented to M*1024), default values of 
 **_n_epochs_** and **_validation_freq_** are 10 and 1, respectively. For a dataset of size 10240 (which gets augmented to M*10240), default values of 
**_n_epochs_** and **_validation_freq_** are 5 and 1, respectively.

## Benchmark methods
The following knobs control the simulation settings of HBF and DBF beamformings.

    gather_data_for_runnig_Sohrabis_and_DBF_beamformer = 'no' or 'yes
    evaluate_sohrabi = 'no' or 'yes
    evaluate_DBF = 'no' or 'yes

Notice that the mentioned beamformings
should be optimized for the effective phase-noised channel for proper comparison with the proposed method. Thus, the 
channel dataset (H) must be transformed using the phase noise samples to H_tilde_0 so that the mentioned beamformings
can be optimized
using the proper channel. If **_gather_data_for_runnig_Sohrabis_and_DBF_beamformer_** is set to 'yes', the code will
produce and save the adjusted phase-noised channel dataset to be used in HBF and DBF optimizers. Once the dataset is 
prepared, HBF and DBF optimizers (**NOT** a part of this repository) can be run separately.

DBF and Sohrabi cannot be evaluated at the same time because in DBF, **_N_b_rf_** should be set to **_N_b_a_** and 
**_N_u_rf_** should be set to **_N_u_a_** which is not the case for HBF. So, if you want to evaluate both, you have to 
do it one at a time.

## Dateset settings
The following knobs control the dataset settings. For datasets larger than 2Gbits it is recommended to 
shard the dataset into multiple files and load them one by one. This is done by setting **_is_data_segmented_** to 
'yes'. The number of shards is controlled by **_train_data_fragment_size_**, (i.e. number of shards =
train_dataset_size/train_data_fragment_size). The default value of 
**_train_data_fragment_size_** is equal to **_train_dataset_size_** which results in one shard. For large datasets, it 
is recommended to set **_train_data_fragment_size_** to a number that results in a size less than 2Gbits.

    is_data_segmented = 'no'
    create_DS_phase_noised = 'yes'
    train_dataset_size = 1024
    train_data_fragment_size = train_dataset_size
    test_dataset_size = 128
    test_data_fragment_size = test_dataset_size
    eval_dataset_size = 128
    eval_data_fragment_size = eval_dataset_size

Same is true for the **_test_dataset_size_** and **_eval_dataset_size_**. The validation dataset is used for adaptive 
learning rate adjustment. During the training the size of the **_eval_dataset_size_** is rather small (e.g. 128 or 256)
to avoid long waiting time and **_eval_dataset_size_** is irrelevant for the training. However, during the evaluation
the size of **_eval_dataset_size_** is set to a large number (e.g. 1024) to get a more accurate evaluation and the 
**_train_dataset_size_** and **_test_dataset_size_** are irrelevant.

## Optimization parameters
The following knobs control the optimization parameters. The following defualt values are recommended for a wide range 
of small and large SU and MU MIMO scenarios. Small batch size is a very good means to prevent overfitting thanks to the 
noisy gradient it creates. Also, **_gradient_norm_clipper_post_** is set to 1.0 to prevent the gradient from exploding
and creating NULL results. Learning rate adjustment is done based on the validation loss through the 
Reduce LR On Plateau policy [https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau] . In the 
mentioned policy, the learning rate is reduced by a factor of **_ReduceLROnPlateau_decay_rate_** when the validation 
loss does not improve for **_ReduceLROnPlateau_patience_** epochs until it hits ReduceLROnPlateau_min_lr = 1e-7. 

    L_rate_initial = 6e-5
    BATCHSIZE = 8
    gradient_norm_clipper_pre = 1. 
    gradient_norm_clipper_post = 1.
    ReduceLROnPlateau_decay_rate = 0.5
    ReduceLROnPlateau_patience = 1
    ReduceLROnPlateau_min_lr = 1e-7

##Neural network parameters
The following knobs control the neural network parameters. The default values are recommended for a wide range of small
and large SU and MU MIMO scenarios. For number of antennas larger than 32, it is recommended to set
**_N_u_a_strides_l1_** and/or **_N_u_a_strides_l2_** and/or **_N_b_a_strides_l1_** and/or 
**_N_b_a_strides_l2_** to 2 to reduce the complexity. These strides are applied to the input of the first and second
convolutional layers. Notice that this repo cannot handle the case where the number of antennas is not divisible by the
their corresponding strides.

**_subcarrier_strides_l1_** should always be set to 1 as the subcarrier domain is already limitted to a small set of
influencial subcarriers and setting it to any value larger than 1 will result in a decline in performance. 

The number of **_convolutional_filters_** (network's width) is maintained throughout the network except for the layers 
that use maxpooling. In those layers the number of filters is multiplied by the same value that is used for maxpooling.

**_n_common_layers_** is the number of common layers in the network that branches out to V_D, V_rf, W_D, and W_rf. 
Notice that each layer is actually an AdaSE-ResNet block which consists of two convolutional layers and two 
fully-connected layers. Maxpoolings and reshape modules are not considered layers as we only refer to trainable 
modules as neural network layers.

**_n_D_and_RF_layers_** is the depth of the V_D, V_rf, W_D, and W_rf branches.

**_n_post_Tconv_processing_** is number of Tconv layers after the common layers to compensate for the maxpooling layers. 

LLR-learning module is not implemented in this repository but the knob **_n_llr_DNN_learner_** is provided for future 
use.

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
    n_llr_DNN_learner =2 # not used

## MIMO-OFDM
The following knobs control the MIMO-OFDM parameters. 
- **_N_s_** is the number of streams.
- **_N_u_** is the number of users.
- **_N_b_a** is the number of BS antennas.
- **_N_u_a** is the number of user antennas.
- **_N_b_rf_** is the number of BS RF chains.
- **_N_u_rf_** is the number of user RF chains.
- **_K_** is the number of subcarriers.
- **_K_prime_** = is size of the influencial subcarriers set
- **_PTRS_seperation_** is the separation between PTRS subcarriers. I abandoned the idea of refreshing RX CSI using PTRS (so it is not used)
- **_SNR_** is the SNR in dB. 
- **_P_** is the power of the transmitted signal.
- **_sigma2_** is the noise variance. 
- **_apply_channel_est_error_** If set to True, the channel estimation 
error is added to the channel estimation. 
- **_channel_est_err_mse_per_element_dB_** is the noise to average channel-element power ratio in dB. It is irrelevent 
if **_apply_channel_est_error_** is set to False.


## Phase noise parameters
The following knobs control the phase noise parameters.
- **_CSIRSPeriod_** is the CSIRS period in number of OFDM symbols.
- **_f_0, L, and fs_** control the phase noise strength through math.sqrt(4.0 * math.pi ** 2 * f_0 ** 2 * 10 ** (L / 10.) / fs)
- **_phase_noise_recorded_** if set to 'yes' uses the recorded lab-measured phase noise. If set to 'no' generates
Wiener phase noise
- **_CLO_or_ILO_** if set to 'ILO' uses the ILO phase noise. If set to 'CLO' uses the CLO phase noise.

## Simulation parameters
The following knobs control the simulation parameters in the inference phase.
- For faster simulation, the trained network is only tested for a subset of the symbols in the CSIRS period. 
The size of the mentioned subset is controled by a ration between 0 and 1 named 
**_sampling_ratio_time_domain_keep_** 
- **_influencial_subcarriers_set_size_** is the size of the influencial subcarriers set. 