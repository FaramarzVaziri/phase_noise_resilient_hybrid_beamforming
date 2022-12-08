import math

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, Conv1D, Conv2D, Reshape, MaxPooling3D, BatchNormalization, Input, \
    Flatten, Dropout, AveragePooling3D, AveragePooling2D, \
    Activation, Add, Conv3DTranspose, Conv1DTranspose, Conv2DTranspose, Multiply, Lambda, Embedding, Concatenate, \
    MaxPooling3D, MaxPooling2D, MaxPooling1D


class ResNet_class(tf.keras.Model):  #
    def __init__(self, filters, kernel_size, strides, dilation, trainable):
        super(ResNet_class, self).__init__()
        self.bn0 = BatchNormalization(trainable=trainable)
        self.conv1 = Conv3D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn1 = BatchNormalization(trainable=trainable)

        self.conv2 = Conv3D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn2 = BatchNormalization(trainable=trainable)

        self.act = Activation('relu')

        # residual connection
        self.res_con = Conv3D(filters=filters,
                              kernel_size=(1, 1, 1),
                              strides=(1, 1, 1),
                              activation=None,
                              padding='same',
                              dilation_rate=dilation,
                              trainable=trainable)

        self.add = Add()

    def call(self, input_tensor):
        y = self.conv1(input_tensor)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        z = self.add([y, self.res_con(input_tensor)])
        # z = self.act(z)
        return z


class ns_adaptive_module_class(tf.keras.Model):  #
    def __init__(self, n1, n2, n3, n_chan, trainable_ns):
        super(ns_adaptive_module_class, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n_chan = n_chan

        self.AP = MaxPooling3D(pool_size=(n1, n2, n3))

        self.FC_mul_1 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.FC_mul_2 = Dense(units=n_chan, activation='sigmoid', use_bias=True, trainable=trainable_ns)

        self.cat = Concatenate(axis=1)

    def call(self, input_tensor, ns):
        y = self.AP(input_tensor)
        y = self.cat([ns, tf.squeeze(y)])

        m = self.FC_mul_1(y)
        m = self.FC_mul_2(m)
        m = tf.tile(tf.reshape(m, shape=[-1, 1, 1, 1, self.n_chan]), multiples=[1, self.n1, self.n2, self.n3, 1])

        z = tf.multiply(input_tensor, m)
        # z = tf.add(z, b)
        return z


class AdaSE_ResNet_class_D(tf.keras.Model):  #
    def __init__(self, filters, kernel_size, strides, dilation, trainable, n1, n2, n3, use_attention):
        super(AdaSE_ResNet_class_D, self).__init__()
        self.bn0 = BatchNormalization(trainable=trainable)
        self.conv1 = Conv3D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn1 = BatchNormalization(trainable=trainable)

        self.conv2 = Conv3D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn2 = BatchNormalization(trainable=trainable)

        self.act = Activation('relu')

        # residual connection
        self.res_con = Conv3D(filters=filters,
                              kernel_size=(1, 1, 1),
                              strides=(1, 1, 1),
                              activation=None,
                              padding='same',
                              dilation_rate=dilation,
                              trainable=trainable)

        # SE part
        self.use_attention = use_attention
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n_chan = filters
        self.AP = AveragePooling3D(pool_size=(n1, n2, n3))
        self.FC_1 = Dense(units=filters, activation='relu', trainable=trainable)
        self.FC_2 = Dense(units=2, activation='relu', trainable=trainable)
        # self.FC_3 = Dense(units=int(filters / 8), activation='relu', trainable=trainable)
        # self.FC_4 = Dense(units=int(filters / 8), activation='relu', trainable=trainable)
        # self.FC_5 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_6 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_7 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_8 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_9 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_10 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_11 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_12 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_13 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_14 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_15 = Dense(units=filters, activation='relu', trainable=trainable)
        self.FC_16 = Dense(units=filters, activation='sigmoid', use_bias=False, trainable=trainable)
        self.cat = Concatenate(axis=1)

        self.add = Add()

    def call(self, input_tensor, ns):
        y = self.conv1(input_tensor)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)

        # SE part
        w = self.AP(y)
        w = self.cat([ns, tf.reshape(w, shape=[-1, self.n_chan])])
        w = self.FC_1(w)
        w = self.FC_2(w)
        # w = self.FC_3(w)
        # w = self.FC_4(w)
        # w = self.FC_5(w)
        # w = self.FC_6(w)
        # w = self.FC_7(w)
        # w = self.FC_8(w)
        # w = self.FC_9(w)
        # w = self.FC_10(w)
        # w = self.FC_11(w)
        # w = self.FC_12(w)
        # w = self.FC_13(w)
        # w = self.FC_14(w)
        # w = self.FC_15(w)
        w = self.FC_16(w)

        w = tf.tile(tf.reshape(w, shape=[-1, 1, 1, 1, self.n_chan]), multiples=[1, self.n1, self.n2, self.n3, 1])

        if self.use_attention == 'yes':
            z = tf.multiply(y, w)
        else:
            z = y
        z = self.add([z, self.res_con(input_tensor)])
        return z


class AdaSE_ResNet_class_RF(tf.keras.Model):  #
    def __init__(self, filters, kernel_size, strides, dilation, trainable, n1, n2, use_attention):
        super(AdaSE_ResNet_class_RF, self).__init__()
        self.bn0 = BatchNormalization(trainable=trainable)
        self.conv1 = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn1 = BatchNormalization(trainable=trainable)

        self.conv2 = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn2 = BatchNormalization(trainable=trainable)

        self.act = Activation('relu')

        # residual connection
        self.res_con = Conv2D(filters=filters,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              activation=None,
                              padding='same',
                              dilation_rate=dilation,
                              trainable=trainable)

        # SE part
        self.use_attention = use_attention
        self.n1 = n1
        self.n2 = n2
        self.n_chan = filters
        self.AP = AveragePooling2D(pool_size=(n1, n2))
        self.FC_1 = Dense(units=filters, activation='relu', trainable=trainable)
        self.FC_2 = Dense(units=2, activation='relu', trainable=trainable)
        # self.FC_3 = Dense(units=int(filters / 8), activation='relu', trainable=trainable)
        # self.FC_4 = Dense(units=int(filters / 8), activation='relu', trainable=trainable)
        # self.FC_5 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_6 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_7 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_8 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_9 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_10 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_11 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_12 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_13 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_14 = Dense(units=filters, activation='relu', trainable=trainable)
        # self.FC_15 = Dense(units=filters, activation='relu', trainable=trainable)
        self.FC_16 = Dense(units=filters, activation='sigmoid', use_bias=False, trainable=trainable)
        self.cat = Concatenate(axis=1)  # channels first

        self.add = Add()

    def call(self, input_tensor, ns):
        y = self.conv1(input_tensor)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)

        # SE part
        w = self.AP(y)
        w = self.cat([ns, tf.reshape(w, shape=[-1, self.n_chan])])

        w = self.FC_1(w)
        w = self.FC_2(w)
        # w = self.FC_3(w)
        # w = self.FC_4(w)
        # w = self.FC_5(w)
        # w = self.FC_6(w)
        # w = self.FC_7(w)
        # w = self.FC_8(w)
        # w = self.FC_9(w)
        # w = self.FC_10(w)
        # w = self.FC_11(w)
        # w = self.FC_12(w)
        # w = self.FC_13(w)
        # w = self.FC_14(w)
        # w = self.FC_15(w)
        w = self.FC_16(w)
        w = tf.tile(tf.reshape(w, shape=[-1, 1, 1, self.n_chan]), multiples=[1, self.n1, self.n2, 1])

        if self.use_attention == 'yes':
            z = tf.multiply(y, w)
        else:
            z = y
        z = self.add([z, self.res_con(input_tensor)])
        return z


class CNN_model_class():
    def __init__(self, setup):
        self.setup = setup
        super(CNN_model_class, self).__init__()

    def CNN_transmiter_AdaSE_ResNet(self, trainable_csi, layer_name):
        kernels_D_start_and_end = [min(self.setup.K_prime, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_b_a, self.setup.convolutional_kernels) + self.setup.extra_kernel]
        kernels_RF = [min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels),
                      min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        kernels_D = [min(self.setup.K_prime, self.setup.convolutional_kernels),
                     min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels),
                     min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        csi = Input(shape=(self.setup.K_prime, self.setup.N_u_a * self.setup.Nue, self.setup.N_b_a, 2), batch_size=self.setup.BATCHSIZE)
        ns = Input(shape=[1], batch_size=self.setup.BATCHSIZE)

        # common path
        AP_1 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l1, self.setup.N_u_a_strides_l1, self.setup.N_b_a_strides_l1))
        AP_2 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l2, self.setup.N_u_a_strides_l2, self.setup.N_b_a_strides_l2))

        AdaSE_block_common_branch = []

        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D_start_and_end,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_a * self.setup.Nue,
                                 n3=self.setup.N_b_a,
                                 use_attention=self.setup.use_attention))
        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a * self.setup.Nue / (
                                     self.setup.N_u_a_strides_l1)),
                                 n3=round(self.setup.N_b_a / (
                                     self.setup.N_b_a_strides_l1)),
                                 use_attention=self.setup.use_attention))

        for i in range(2, self.setup.n_common_layers, 1):
            AdaSE_block_common_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                                  kernel_size=kernels_D,
                                                                  strides=self.setup.convolutional_strides,
                                                                  dilation=self.setup.convolutional_dilation,
                                                                  trainable=trainable_csi,
                                                                  n1=round(self.setup.K_prime / (
                                                                          self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                                  n2=round(self.setup.N_u_a * self.setup.Nue / (
                                                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                  n3=round(self.setup.N_b_a / (
                                                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                  use_attention=self.setup.use_attention))

        # V_D path
        AdaSE_block_D_branch = []
        # ns_adaptive_module_D =[]
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_block_D_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                             kernel_size=kernels_D,
                                                             strides=self.setup.convolutional_strides,
                                                             dilation=self.setup.convolutional_dilation,
                                                             trainable=trainable_csi,
                                                             n1=round(self.setup.K_prime / (
                                                                     self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                             n2=round(self.setup.N_u_a * self.setup.Nue / (
                                                                     self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                             n3=round(self.setup.N_b_a / (
                                                                     self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                             use_attention=self.setup.use_attention))

        Tconv_D_1 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                    kernel_size=[self.setup.subcarrier_strides_l2, 1, 1],
                                    strides=(self.setup.subcarrier_strides_l2, 1, 1),
                                    padding='valid')
        AdaSE_ResNet_block_D_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a * self.setup.Nue / (
                                         self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                 n3=round(self.setup.N_b_a / (
                                         self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                 use_attention=self.setup.use_attention))
        Tconv_D_2 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                    kernel_size=[self.setup.subcarrier_strides_l1, 1, 1],
                                    strides=[self.setup.subcarrier_strides_l1, 1, 1],
                                    padding='valid')
        current_size_dim_Nua = round(self.setup.N_u_a * self.setup.Nue / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))

        if self.setup.Nue == 1:
            AP_D_branch_end = MaxPooling3D(pool_size=(1, 1, round(current_size_dim_Nba / self.setup.N_s)))
        else:
            AP_D_branch_end = MaxPooling3D(
                pool_size=(1, round(current_size_dim_Nba / self.setup.N_b_rf), round(current_size_dim_Nba / self.setup.N_s)))

        reshaper_D = Reshape(target_shape=[self.setup.K_prime, self.setup.N_b_rf * self.setup.Nue, self.setup.N_s, -1])
        for i in range(1, self.setup.n_post_Tconv_processing - 1, 1):
            AdaSE_ResNet_block_D_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                     kernel_size=kernels_D,
                                     strides=self.setup.convolutional_strides,
                                     dilation=self.setup.convolutional_dilation,
                                     trainable=trainable_csi,
                                     n1=self.setup.K_prime,
                                     n2=self.setup.N_b_rf * self.setup.Nue,
                                     n3=self.setup.N_s,
                                     use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=2,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_b_rf * self.setup.Nue,
                                 n3=self.setup.N_s,
                                 use_attention=self.setup.use_attention))
        reshaper_D_end = Reshape(target_shape=[self.setup.Nue, self.setup.K_prime, self.setup.N_b_rf, self.setup.N_s, -1])

        # V_RF path-------------------
        AP_rf = MaxPooling3D(pool_size=(round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                        self.setup.Nue,
                                        1))
        reshaper_RF_begin = Reshape(
            target_shape=[round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])

        AdaSE_ResNet_block_RF_branch = []
        # ns_adaptive_module_RF = []
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_ResNet_block_RF_branch.append(AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                                                      kernel_size=kernels_RF,
                                                                      strides=self.setup.convolutional_strides,
                                                                      dilation=self.setup.convolutional_dilation,
                                                                      trainable=trainable_csi,
                                                                      n1=round(self.setup.N_u_a / (
                                                                              self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                      n2=round(self.setup.N_b_a / (
                                                                              self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                      use_attention=self.setup.use_attention))

        Tconv_V_RF_1 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[1, self.setup.N_b_a_strides_l2],
                                       strides=(1, self.setup.N_b_a_strides_l2),
                                       padding='valid')
        AdaSE_ResNet_block_RF_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=round(self.setup.N_u_a / (
                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                  n2=round(self.setup.N_b_a / (
                                      self.setup.N_b_a_strides_l1)),
                                  use_attention=self.setup.use_attention))

        Tconv_V_RF_2 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[1, self.setup.N_b_a_strides_l1],
                                       strides=[1, self.setup.N_b_a_strides_l1],
                                       padding='valid')

        current_size_dim_Nua = round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_dim_Nba = self.setup.N_b_a
        for i in range(0, self.setup.n_post_Tconv_processing - 2, 1):
            AdaSE_ResNet_block_RF_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                      kernel_size=kernels_RF,
                                      strides=self.setup.convolutional_strides,
                                      dilation=self.setup.convolutional_dilation,
                                      trainable=trainable_csi,
                                      n1=current_size_dim_Nua,
                                      n2=current_size_dim_Nba,
                                      use_attention=self.setup.use_attention))
        AdaSE_ResNet_block_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=1,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=current_size_dim_Nua,
                                  n2=current_size_dim_Nba,
                                  use_attention=self.setup.use_attention))

        AP_RF_branch_end = MaxPooling2D(pool_size=(current_size_dim_Nua, 1))
        reshaper_RF = Reshape(target_shape=[self.setup.N_b_a])

        # connections

        x = AdaSE_block_common_branch[0](csi, ns)
        x = AP_1(x)
        x = AdaSE_block_common_branch[1](x, ns)
        x = AP_2(x)

        for i in range(2, self.setup.n_common_layers, 1):
            x = AdaSE_block_common_branch[i](x, ns)

        # V_D path
        vd = AdaSE_block_D_branch[0](x, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vd = AdaSE_block_D_branch[i](vd, ns)

        vd = Tconv_D_1(vd)
        vd = AdaSE_ResNet_block_D_branch_post_Tconv_processing[0](vd, ns)
        vd = Tconv_D_2(vd)
        vd = AP_D_branch_end(vd)
        vd = reshaper_D(vd)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vd = AdaSE_ResNet_block_D_branch_post_Tconv_processing[i](vd, ns)
        vd = reshaper_D_end(vd)

        # V_RF path
        vrf = AP_rf(x)
        vrf = reshaper_RF_begin(vrf)
        vrf = AdaSE_ResNet_block_RF_branch[0](vrf, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vrf = AdaSE_ResNet_block_RF_branch[i](vrf, ns)

        vrf = Tconv_V_RF_1(vrf)
        vrf = AdaSE_ResNet_block_RF_branch_post_Tconv_processing[0](vrf, ns)
        vrf = Tconv_V_RF_2(vrf)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vrf = AdaSE_ResNet_block_RF_branch_post_Tconv_processing[i](vrf, ns)

        vrf = AP_RF_branch_end(vrf)
        vrf = reshaper_RF(vrf)

        func_model = Model(inputs=[csi, ns], outputs=[vd, vrf], name=layer_name)
        return func_model

    def CNN_receiver_AdaSE_ResNet(self, trainable_csi, layer_name):
        kernels_D_start_and_end = [min(self.setup.K_prime, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_u_a, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_b_a, self.setup.convolutional_kernels) + self.setup.extra_kernel]
        kernels_RF = [min(self.setup.N_u_a, self.setup.convolutional_kernels),
                      min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        kernels_D = [min(self.setup.K_prime, self.setup.convolutional_kernels),
                     min(self.setup.N_u_a, self.setup.convolutional_kernels),
                     min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        csi = Input(shape=(self.setup.K_prime, self.setup.N_u_a, self.setup.N_b_a, 2), batch_size=self.setup.BATCHSIZE)
        ns = Input(shape=[1], batch_size=self.setup.BATCHSIZE)

        # common path
        AP_1 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l1, self.setup.N_u_a_strides_l1, self.setup.N_b_a_strides_l1))
        AP_2 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l2, self.setup.N_u_a_strides_l2, self.setup.N_b_a_strides_l2))

        AdaSE_block_common_branch = []

        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D_start_and_end,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_a,
                                 n3=self.setup.N_b_a,
                                 use_attention=self.setup.use_attention))
        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a / (
                                     self.setup.N_u_a_strides_l1)),
                                 n3=round(self.setup.N_b_a / (
                                     self.setup.N_b_a_strides_l1)),
                                 use_attention=self.setup.use_attention))

        for i in range(2, self.setup.n_common_layers, 1):
            AdaSE_block_common_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                                  kernel_size=kernels_D,
                                                                  strides=self.setup.convolutional_strides,
                                                                  dilation=self.setup.convolutional_dilation,
                                                                  trainable=trainable_csi,
                                                                  n1=round(self.setup.K_prime / (
                                                                          self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                                  n2=round(self.setup.N_u_a / (
                                                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                  n3=round(self.setup.N_b_a / (
                                                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                  use_attention=self.setup.use_attention))

        # V_D path
        AdaSE_block_D_branch = []
        # ns_adaptive_module_D =[]
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_block_D_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                             kernel_size=kernels_D,
                                                             strides=self.setup.convolutional_strides,
                                                             dilation=self.setup.convolutional_dilation,
                                                             trainable=trainable_csi,
                                                             n1=round(self.setup.K_prime / (
                                                                     self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                             n2=round(self.setup.N_u_a / (
                                                                     self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                             n3=round(self.setup.N_b_a / (
                                                                     self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                             use_attention=self.setup.use_attention))

        Tconv_D_1 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                    kernel_size=[self.setup.subcarrier_strides_l2, 1, 1],
                                    strides=(self.setup.subcarrier_strides_l2, 1, 1),
                                    padding='valid')
        AdaSE_ResNet_block_D_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a / (
                                         self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                 n3=round(self.setup.N_b_a / (
                                         self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                 use_attention=self.setup.use_attention))
        Tconv_D_2 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                    kernel_size=[self.setup.subcarrier_strides_l1, 1, 1],
                                    strides=[self.setup.subcarrier_strides_l1, 1, 1],
                                    padding='valid')
        current_size_dim_Nua = round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))

        AP_D_branch_end = MaxPooling3D(
            pool_size=(1, int(current_size_dim_Nua / self.setup.N_u_rf), int(current_size_dim_Nua / self.setup.N_s)))
        reshaper_D = Reshape(target_shape=[self.setup.K_prime, self.setup.N_u_rf, self.setup.N_s, -1])
        for i in range(1, self.setup.n_post_Tconv_processing - 1, 1):
            AdaSE_ResNet_block_D_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                     kernel_size=kernels_D,
                                     strides=self.setup.convolutional_strides,
                                     dilation=self.setup.convolutional_dilation,
                                     trainable=trainable_csi,
                                     n1=self.setup.K_prime,
                                     n2=self.setup.N_u_rf,
                                     n3=self.setup.N_s,
                                     use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=2,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_rf,
                                 n3=self.setup.N_s,
                                 use_attention=self.setup.use_attention))

        # V_RF path-------------------
        AP_rf = MaxPooling3D(pool_size=(round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                        1,
                                        1))
        reshaper_RF_begin = Reshape(
            target_shape=[round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])

        AdaSE_ResNet_block_RF_branch = []
        # ns_adaptive_module_RF = []
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_ResNet_block_RF_branch.append(AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                                                      kernel_size=kernels_RF,
                                                                      strides=self.setup.convolutional_strides,
                                                                      dilation=self.setup.convolutional_dilation,
                                                                      trainable=trainable_csi,
                                                                      n1=round(self.setup.N_u_a / (
                                                                              self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                      n2=round(self.setup.N_b_a / (
                                                                              self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                      use_attention=self.setup.use_attention))

        Tconv_V_RF_1 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[self.setup.N_u_a_strides_l2, 1],
                                       strides=(self.setup.N_u_a_strides_l2, 1),
                                       padding='valid')

        AdaSE_ResNet_block_RF_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=round(self.setup.N_u_a / (
                                      self.setup.N_u_a_strides_l1)),
                                  n2=round(self.setup.N_b_a / (
                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                  use_attention=self.setup.use_attention))

        Tconv_V_RF_2 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[self.setup.N_u_a_strides_l1, 1],
                                       strides=[self.setup.N_u_a_strides_l1, 1],
                                       padding='valid')

        current_size_dim_Nua = self.setup.N_u_a
        current_size_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))
        for i in range(0, self.setup.n_post_Tconv_processing - 2, 1):
            AdaSE_ResNet_block_RF_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                      kernel_size=kernels_RF,
                                      strides=self.setup.convolutional_strides,
                                      dilation=self.setup.convolutional_dilation,
                                      trainable=trainable_csi,
                                      n1=current_size_dim_Nua,
                                      n2=current_size_dim_Nba,
                                      use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_RF_branch_post_Tconv_processing.append(AdaSE_ResNet_class_RF(filters=1,
                                                                                        kernel_size=kernels_RF,
                                                                                        strides=self.setup.convolutional_strides,
                                                                                        dilation=self.setup.convolutional_dilation,
                                                                                        trainable=trainable_csi,
                                                                                        n1=current_size_dim_Nua,
                                                                                        n2=current_size_dim_Nba,
                                                                                        use_attention=self.setup.use_attention))

        AP_RF_branch_end = MaxPooling2D(pool_size=(1, current_size_dim_Nba))
        reshaper_RF = Reshape(target_shape=[self.setup.N_u_a])

        # connections

        x = AdaSE_block_common_branch[0](csi, ns)
        x = AP_1(x)
        x = AdaSE_block_common_branch[1](x, ns)
        x = AP_2(x)

        for i in range(2, self.setup.n_common_layers, 1):
            x = AdaSE_block_common_branch[i](x, ns)

        # V_D path
        vd = AdaSE_block_D_branch[0](x, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vd = AdaSE_block_D_branch[i](vd, ns)

        vd = Tconv_D_1(vd)
        vd = AdaSE_ResNet_block_D_branch_post_Tconv_processing[0](vd, ns)
        vd = Tconv_D_2(vd)
        vd = AP_D_branch_end(vd)
        vd = reshaper_D(vd)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vd = AdaSE_ResNet_block_D_branch_post_Tconv_processing[i](vd, ns)

        # V_RF path
        vrf = AP_rf(x)
        vrf = reshaper_RF_begin(vrf)
        vrf = AdaSE_ResNet_block_RF_branch[0](vrf, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vrf = AdaSE_ResNet_block_RF_branch[i](vrf, ns)

        vrf = Tconv_V_RF_1(vrf)
        vrf = AdaSE_ResNet_block_RF_branch_post_Tconv_processing[0](vrf, ns)
        vrf = Tconv_V_RF_2(vrf)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vrf = AdaSE_ResNet_block_RF_branch_post_Tconv_processing[i](vrf, ns)

        vrf = AP_RF_branch_end(vrf)
        vrf = reshaper_RF(vrf)

        func_model = Model(inputs=[csi, ns], outputs=[vd, vrf], name=layer_name)
        return func_model

    def CNN_transceiver_AdaSE_ResNet_no_binding(self, trainable_csi, layer_name):
        kernels_D_start_and_end = [min(self.setup.K_prime, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_b_a, self.setup.convolutional_kernels) + self.setup.extra_kernel]
        kernels_RF = [min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels),
                      min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        kernels_D = [min(self.setup.K_prime, self.setup.convolutional_kernels),
                     min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels),
                     min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        csi = Input(shape=(self.setup.K_prime, self.setup.N_u_a * self.setup.Nue, self.setup.N_b_a, 2), batch_size=self.setup.BATCHSIZE)
        ns = Input(shape=[1], batch_size=self.setup.BATCHSIZE)

        # common path __________________________________________________________________________________________________
        AP_1 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l1, self.setup.N_u_a_strides_l1, self.setup.N_b_a_strides_l1))
        AP_2 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l2, self.setup.N_u_a_strides_l2, self.setup.N_b_a_strides_l2))

        AdaSE_block_common_branch = []

        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D_start_and_end,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_a * self.setup.Nue,
                                 n3=self.setup.N_b_a,
                                 use_attention=self.setup.use_attention))
        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a * self.setup.Nue / (
                                     self.setup.N_u_a_strides_l1)),
                                 n3=round(self.setup.N_b_a / (
                                     self.setup.N_b_a_strides_l1)),
                                 use_attention=self.setup.use_attention))

        for i in range(2, self.setup.n_common_layers, 1):
            AdaSE_block_common_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                                  kernel_size=kernels_D,
                                                                  strides=self.setup.convolutional_strides,
                                                                  dilation=self.setup.convolutional_dilation,
                                                                  trainable=trainable_csi,
                                                                  n1=round(self.setup.K_prime / (
                                                                          self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                                  n2=round(self.setup.N_u_a * self.setup.Nue / (
                                                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                  n3=round(self.setup.N_b_a / (
                                                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                  use_attention=self.setup.use_attention))

        # V_D path __________________________________________________________________________________________________
        AdaSE_block_V_D_branch = []
        # ns_adaptive_module_D =[]
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_block_V_D_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                               kernel_size=kernels_D,
                                                               strides=self.setup.convolutional_strides,
                                                               dilation=self.setup.convolutional_dilation,
                                                               trainable=trainable_csi,
                                                               n1=round(self.setup.K_prime / (
                                                                       self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                               n2=round(self.setup.N_u_a * self.setup.Nue / (
                                                                       self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                               n3=round(self.setup.N_b_a / (
                                                                       self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                               use_attention=self.setup.use_attention))

        Tconv_V_D_1 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l2, 1, 1],
                                      strides=(self.setup.subcarrier_strides_l2, 1, 1),
                                      padding='valid')
        AdaSE_ResNet_block_V_D_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_V_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a * self.setup.Nue / (
                                         self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                 n3=round(self.setup.N_b_a / (
                                         self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                 use_attention=self.setup.use_attention))
        Tconv_V_D_2 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l1, 1, 1],
                                      strides=[self.setup.subcarrier_strides_l1, 1, 1],
                                      padding='valid')
        current_size_V_dim_Nua = round(self.setup.N_u_a * self.setup.Nue / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_V_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))

        if self.setup.Nue == 1:
            AP_V_D_branch_end = MaxPooling3D(pool_size=(1, 1, round(current_size_V_dim_Nba / self.setup.N_s)))
        else:
            AP_V_D_branch_end = MaxPooling3D(
                pool_size=(1, round(current_size_V_dim_Nba / self.setup.N_b_rf), round(current_size_V_dim_Nba / self.setup.N_s)))

        reshaper_V_D = Reshape(target_shape=[self.setup.K_prime, self.setup.N_b_rf * self.setup.Nue, self.setup.N_s, -1])
        for i in range(1, self.setup.n_post_Tconv_processing - 1, 1):
            AdaSE_ResNet_block_V_D_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                     kernel_size=kernels_D,
                                     strides=self.setup.convolutional_strides,
                                     dilation=self.setup.convolutional_dilation,
                                     trainable=trainable_csi,
                                     n1=self.setup.K_prime,
                                     n2=self.setup.N_b_rf * self.setup.Nue,
                                     n3=self.setup.N_s,
                                     use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_V_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=2,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_b_rf * self.setup.Nue,
                                 n3=self.setup.N_s,
                                 use_attention=self.setup.use_attention))
        reshaper_V_D_end = Reshape(target_shape=[self.setup.Nue, self.setup.K_prime, self.setup.N_b_rf, self.setup.N_s, -1])

        # V_RF path __________________________________________________________________________________________________
        AP_V_rf = MaxPooling3D(
            pool_size=(round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                       self.setup.Nue,
                       1))
        reshaper_V_RF_begin = Reshape(
            target_shape=[round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])

        AdaSE_ResNet_block_V_RF_branch = []
        # ns_adaptive_module_RF = []
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_ResNet_block_V_RF_branch.append(AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                                                        kernel_size=kernels_RF,
                                                                        strides=self.setup.convolutional_strides,
                                                                        dilation=self.setup.convolutional_dilation,
                                                                        trainable=trainable_csi,
                                                                        n1=round(self.setup.N_u_a / (
                                                                                self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                        n2=round(self.setup.N_b_a / (
                                                                                self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                        use_attention=self.setup.use_attention))

        Tconv_V_RF_1 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[1, self.setup.N_b_a_strides_l2],
                                       strides=(1, self.setup.N_b_a_strides_l2),
                                       padding='valid')
        AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=round(self.setup.N_u_a / (
                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                  n2=round(self.setup.N_b_a / (
                                      self.setup.N_b_a_strides_l1)),
                                  use_attention=self.setup.use_attention))

        Tconv_V_RF_2 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[1, self.setup.N_b_a_strides_l1],
                                       strides=[1, self.setup.N_b_a_strides_l1],
                                       padding='valid')

        current_size_V_dim_Nua = round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_V_dim_Nba = self.setup.N_b_a
        for i in range(0, self.setup.n_post_Tconv_processing - 2, 1):
            AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                      kernel_size=kernels_RF,
                                      strides=self.setup.convolutional_strides,
                                      dilation=self.setup.convolutional_dilation,
                                      trainable=trainable_csi,
                                      n1=current_size_V_dim_Nua,
                                      n2=current_size_V_dim_Nba,
                                      use_attention=self.setup.use_attention))
        AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=1,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=current_size_V_dim_Nua,
                                  n2=current_size_V_dim_Nba,
                                  use_attention=self.setup.use_attention))

        AP_V_RF_branch_end = MaxPooling2D(pool_size=(current_size_V_dim_Nua, 1))
        reshaper_V_RF = Reshape(target_shape=[self.setup.N_b_a])

        # W_D path __________________________________________________________________________________________________
        AdaSE_block_W_D_branch = []
        # ns_adaptive_module_D =[]
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_block_W_D_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                               kernel_size=kernels_D,
                                                               strides=self.setup.convolutional_strides,
                                                               dilation=self.setup.convolutional_dilation,
                                                               trainable=trainable_csi,
                                                               n1=round(self.setup.K_prime / (
                                                                       self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                               n2=round(self.setup.N_u_a * self.setup.Nue / (
                                                                       self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                               n3=round(self.setup.N_b_a / (
                                                                       self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                               use_attention=self.setup.use_attention))

        Tconv_W_D_1 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l2, 1, 1],
                                      strides=(self.setup.subcarrier_strides_l2, 1, 1),
                                      padding='valid')
        AdaSE_ResNet_block_W_D_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_W_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a * self.setup.Nue / (
                                         self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                 n3=round(self.setup.N_b_a / (
                                         self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                 use_attention=self.setup.use_attention))
        Tconv_W_D_2 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l1, 1, 1],
                                      strides=[self.setup.subcarrier_strides_l1, 1, 1],
                                      padding='valid')
        current_size_W_dim_Nua = round(self.setup.N_u_a * self.setup.Nue / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_W_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))

        AP_W_D_branch_end = MaxPooling3D(
            pool_size=(1, int(current_size_W_dim_Nua / self.setup.N_u_rf), int(current_size_W_dim_Nua / self.setup.N_s)))
        reshaper_W_D = Reshape(target_shape=[self.setup.K_prime, self.setup.Nue * self.setup.N_u_rf, self.setup.N_s, -1])
        for i in range(1, self.setup.n_post_Tconv_processing - 1, 1):
            AdaSE_ResNet_block_W_D_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                     kernel_size=kernels_D,
                                     strides=self.setup.convolutional_strides,
                                     dilation=self.setup.convolutional_dilation,
                                     trainable=trainable_csi,
                                     n1=self.setup.K_prime,
                                     n2=self.setup.N_u_rf * self.setup.Nue,
                                     n3=self.setup.N_s,
                                     use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_W_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=2,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_rf * self.setup.Nue,
                                 n3=self.setup.N_s,
                                 use_attention=self.setup.use_attention))
        reshaper_W_D_end = Reshape(target_shape=[self.setup.Nue, self.setup.K_prime, self.setup.N_u_rf, self.setup.N_s, -1])

        # W_RF path-------------------
        AP_W_rf = MaxPooling3D(
            pool_size=(round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                       1,
                       1))
        reshaper_W_RF_begin = Reshape(
            target_shape=[round(self.setup.N_u_a * self.setup.Nue / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])

        AdaSE_ResNet_block_W_RF_branch = []

        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_ResNet_block_W_RF_branch.append(AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                                                        kernel_size=kernels_RF,
                                                                        strides=self.setup.convolutional_strides,
                                                                        dilation=self.setup.convolutional_dilation,
                                                                        trainable=trainable_csi,
                                                                        n1=round(self.setup.N_u_a * self.setup.Nue / (
                                                                                self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                        n2=round(self.setup.N_b_a / (
                                                                                self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                        use_attention=self.setup.use_attention))

        Tconv_W_RF_1 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[self.setup.N_u_a_strides_l2, 1],
                                       strides=(self.setup.N_u_a_strides_l2, 1),
                                       padding='valid')

        AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=round(self.setup.N_u_a * self.setup.Nue / (
                                      self.setup.N_u_a_strides_l1)),
                                  n2=round(self.setup.N_b_a / (
                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                  use_attention=self.setup.use_attention))

        Tconv_W_RF_2 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[self.setup.N_u_a_strides_l1, 1],
                                       strides=[self.setup.N_u_a_strides_l1, 1],
                                       padding='valid')

        current_size_W_dim_Nua = self.setup.N_u_a * self.setup.Nue
        current_size_W_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))
        for i in range(0, self.setup.n_post_Tconv_processing - 2, 1):
            AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                      kernel_size=kernels_RF,
                                      strides=self.setup.convolutional_strides,
                                      dilation=self.setup.convolutional_dilation,
                                      trainable=trainable_csi,
                                      n1=current_size_W_dim_Nua,
                                      n2=current_size_W_dim_Nba,
                                      use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing.append(AdaSE_ResNet_class_RF(filters=1,
                                                                                          kernel_size=kernels_RF,
                                                                                          strides=self.setup.convolutional_strides,
                                                                                          dilation=self.setup.convolutional_dilation,
                                                                                          trainable=trainable_csi,
                                                                                          n1=current_size_W_dim_Nua,
                                                                                          n2=current_size_W_dim_Nba,
                                                                                          use_attention=self.setup.use_attention))

        AP_W_RF_branch_end = MaxPooling2D(pool_size=(1, current_size_W_dim_Nba))
        reshaper_W_RF = Reshape(target_shape=[self.setup.Nue, self.setup.N_u_a])

        # connections

        x = AdaSE_block_common_branch[0](csi, ns)
        x = AP_1(x)
        x = AdaSE_block_common_branch[1](x, ns)
        x = AP_2(x)

        for i in range(2, self.setup.n_common_layers, 1):
            x = AdaSE_block_common_branch[i](x, ns)

        # V_D path
        vd = AdaSE_block_V_D_branch[0](x, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vd = AdaSE_block_V_D_branch[i](vd, ns)

        vd = Tconv_V_D_1(vd)
        vd = AdaSE_ResNet_block_V_D_branch_post_Tconv_processing[0](vd, ns)
        vd = Tconv_V_D_2(vd)
        vd = AP_V_D_branch_end(vd)
        vd = reshaper_V_D(vd)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vd = AdaSE_ResNet_block_V_D_branch_post_Tconv_processing[i](vd, ns)
        vd = reshaper_V_D_end(vd)

        # V_RF path
        vrf = AP_V_rf(x)
        vrf = reshaper_V_RF_begin(vrf)
        vrf = AdaSE_ResNet_block_V_RF_branch[0](vrf, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vrf = AdaSE_ResNet_block_V_RF_branch[i](vrf, ns)

        vrf = Tconv_V_RF_1(vrf)
        vrf = AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing[0](vrf, ns)
        vrf = Tconv_V_RF_2(vrf)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vrf = AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing[i](vrf, ns)

        vrf = AP_V_RF_branch_end(vrf)
        vrf = reshaper_V_RF(vrf)

        # W_RF path
        wd = AdaSE_block_W_D_branch[0](x, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            wd = AdaSE_block_W_D_branch[i](wd, ns)

        wd = Tconv_W_D_1(wd)
        wd = AdaSE_ResNet_block_W_D_branch_post_Tconv_processing[0](wd, ns)
        wd = Tconv_W_D_2(wd)
        wd = AP_W_D_branch_end(wd)
        wd = reshaper_W_D(wd)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            wd = AdaSE_ResNet_block_W_D_branch_post_Tconv_processing[i](wd, ns)
        wd = reshaper_W_D_end(wd)

        # W_RF path
        wrf = AP_W_rf(x)
        wrf = reshaper_W_RF_begin(wrf)
        wrf = AdaSE_ResNet_block_W_RF_branch[0](wrf, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            wrf = AdaSE_ResNet_block_W_RF_branch[i](wrf, ns)

        wrf = Tconv_W_RF_1(wrf)
        wrf = AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing[0](wrf, ns)
        wrf = Tconv_W_RF_2(wrf)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            wrf = AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing[i](wrf, ns)

        wrf = AP_W_RF_branch_end(wrf)
        wrf = reshaper_W_RF(wrf)

        func_model = Model(inputs=[csi, ns], outputs=[vd, vrf, wd, wrf], name=layer_name)
        return func_model

    def CNN_transceiver_AdaSE_ResNet_with_binding(self, trainable_csi, layer_name):
        kernels_D_start_and_end = [min(self.setup.K_prime, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels) + self.setup.extra_kernel,
                                   min(self.setup.N_b_a, self.setup.convolutional_kernels) + self.setup.extra_kernel]
        kernels_RF = [min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels),
                      min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        kernels_D = [min(self.setup.K_prime, self.setup.convolutional_kernels),
                     min(self.setup.N_u_a * self.setup.Nue, self.setup.convolutional_kernels),
                     min(self.setup.N_b_a, self.setup.convolutional_kernels)]
        csi = Input(shape=(self.setup.K_prime, self.setup.N_u_a * self.setup.Nue, self.setup.N_b_a, 2), batch_size=self.setup.BATCHSIZE)
        ns = Input(shape=[1], batch_size=self.setup.BATCHSIZE)

        # common path __________________________________________________________________________________________________
        AP_1 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l1, self.setup.N_u_a_strides_l1, self.setup.N_b_a_strides_l1))
        AP_2 = MaxPooling3D(pool_size=(self.setup.subcarrier_strides_l2, self.setup.N_u_a_strides_l2, self.setup.N_b_a_strides_l2))

        AdaSE_block_common_branch = []

        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D_start_and_end,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_a * self.setup.Nue,
                                 n3=self.setup.N_b_a,
                                 use_attention=self.setup.use_attention))
        AdaSE_block_common_branch.append(
            AdaSE_ResNet_class_D(filters=int(self.setup.convolutional_filters),
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a * self.setup.Nue / (
                                     self.setup.N_u_a_strides_l1)),
                                 n3=round(self.setup.N_b_a / (
                                     self.setup.N_b_a_strides_l1)),
                                 use_attention=self.setup.use_attention))

        for i in range(2, self.setup.n_common_layers, 1):
            AdaSE_block_common_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                                  kernel_size=kernels_D,
                                                                  strides=self.setup.convolutional_strides,
                                                                  dilation=self.setup.convolutional_dilation,
                                                                  trainable=trainable_csi,
                                                                  n1=round(self.setup.K_prime / (
                                                                          self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                                  n2=round(self.setup.N_u_a * self.setup.Nue / (
                                                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                  n3=round(self.setup.N_b_a / (
                                                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                  use_attention=self.setup.use_attention))

        reshaper_common = Reshape(
            target_shape=[round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                          self.setup.Nue,
                          round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])
        # UE subnetwork ________________________________________________________________________________________________
        # V_D path _____________________________________________________________________________________________________
        AdaSE_block_V_D_branch = []
        # ns_adaptive_module_D =[]
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_block_V_D_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                               kernel_size=kernels_D,
                                                               strides=self.setup.convolutional_strides,
                                                               dilation=self.setup.convolutional_dilation,
                                                               trainable=trainable_csi,
                                                               n1=round(self.setup.K_prime / (
                                                                       self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                               n2=round(self.setup.N_u_a / (
                                                                       self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                               n3=round(self.setup.N_b_a / (
                                                                       self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                               use_attention=self.setup.use_attention))

        Tconv_V_D_1 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l2, 1, 1],
                                      strides=(self.setup.subcarrier_strides_l2, 1, 1),
                                      padding='valid')
        AdaSE_ResNet_block_V_D_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_V_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a / (
                                         self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                 n3=round(self.setup.N_b_a / (
                                         self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                 use_attention=self.setup.use_attention))
        Tconv_V_D_2 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l1, 1, 1],
                                      strides=[self.setup.subcarrier_strides_l1, 1, 1],
                                      padding='valid')
        current_size_V_dim_Nua = round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_V_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))

        if self.setup.Nue == 1:
            AP_V_D_branch_end = MaxPooling3D(pool_size=(1, 1, round(current_size_V_dim_Nba / self.setup.N_s)))
        else:
            AP_V_D_branch_end = MaxPooling3D(
                pool_size=(1, round(current_size_V_dim_Nba / self.setup.N_b_rf), round(current_size_V_dim_Nba / self.setup.N_s)))

        reshaper_V_D = Reshape(target_shape=[self.setup.K_prime, self.setup.N_b_rf, self.setup.N_s, -1])
        for i in range(1, self.setup.n_post_Tconv_processing - 1, 1):
            AdaSE_ResNet_block_V_D_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                     kernel_size=kernels_D,
                                     strides=self.setup.convolutional_strides,
                                     dilation=self.setup.convolutional_dilation,
                                     trainable=trainable_csi,
                                     n1=self.setup.K_prime,
                                     n2=self.setup.N_b_rf,
                                     n3=self.setup.N_s,
                                     use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_V_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=2,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_b_rf,
                                 n3=self.setup.N_s,
                                 use_attention=self.setup.use_attention))
        reshaper_V_D_end = Reshape(target_shape=[self.setup.K_prime, self.setup.N_b_rf, self.setup.N_s, -1])

        # W_D path __________________________________________________________________________________________________
        AdaSE_block_W_D_branch = []
        # ns_adaptive_module_D =[]
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_block_W_D_branch.append(AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                                               kernel_size=kernels_D,
                                                               strides=self.setup.convolutional_strides,
                                                               dilation=self.setup.convolutional_dilation,
                                                               trainable=trainable_csi,
                                                               n1=round(self.setup.K_prime / (
                                                                       self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                                                               n2=round(self.setup.N_u_a / (
                                                                       self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                               n3=round(self.setup.N_b_a / (
                                                                       self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                               use_attention=self.setup.use_attention))

        Tconv_W_D_1 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l2, 1, 1],
                                      strides=(self.setup.subcarrier_strides_l2, 1, 1),
                                      padding='valid')
        AdaSE_ResNet_block_W_D_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_W_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=round(self.setup.K_prime / (
                                     self.setup.subcarrier_strides_l1)),
                                 n2=round(self.setup.N_u_a / (
                                         self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                 n3=round(self.setup.N_b_a / (
                                         self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                 use_attention=self.setup.use_attention))
        Tconv_W_D_2 = Conv3DTranspose(filters=self.setup.convolutional_filters,
                                      kernel_size=[self.setup.subcarrier_strides_l1, 1, 1],
                                      strides=[self.setup.subcarrier_strides_l1, 1, 1],
                                      padding='valid')
        current_size_W_dim_Nua = round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_W_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))

        AP_W_D_branch_end = MaxPooling3D(
            pool_size=(1, int(current_size_W_dim_Nua / self.setup.N_u_rf), int(current_size_W_dim_Nua / self.setup.N_s)))
        reshaper_W_D = Reshape(target_shape=[self.setup.K_prime, self.setup.N_u_rf, self.setup.N_s, -1])
        for i in range(1, self.setup.n_post_Tconv_processing - 1, 1):
            AdaSE_ResNet_block_W_D_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_D(filters=self.setup.convolutional_filters,
                                     kernel_size=kernels_D,
                                     strides=self.setup.convolutional_strides,
                                     dilation=self.setup.convolutional_dilation,
                                     trainable=trainable_csi,
                                     n1=self.setup.K_prime,
                                     n2=self.setup.N_u_rf,
                                     n3=self.setup.N_s,
                                     use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_W_D_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_D(filters=2,
                                 kernel_size=kernels_D,
                                 strides=self.setup.convolutional_strides,
                                 dilation=self.setup.convolutional_dilation,
                                 trainable=trainable_csi,
                                 n1=self.setup.K_prime,
                                 n2=self.setup.N_u_rf,
                                 n3=self.setup.N_s,
                                 use_attention=self.setup.use_attention))
        reshaper_W_D_end = Reshape(target_shape=[self.setup.K_prime, self.setup.N_u_rf, self.setup.N_s, -1])

        # W_RF path-------------------
        AP_W_rf = MaxPooling3D(
            pool_size=(round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                       1,
                       1))
        reshaper_W_RF_begin = Reshape(
            target_shape=[round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])

        AdaSE_ResNet_block_W_RF_branch = []

        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_ResNet_block_W_RF_branch.append(AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                                                        kernel_size=kernels_RF,
                                                                        strides=self.setup.convolutional_strides,
                                                                        dilation=self.setup.convolutional_dilation,
                                                                        trainable=trainable_csi,
                                                                        n1=round(self.setup.N_u_a / (
                                                                                self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                        n2=round(self.setup.N_b_a / (
                                                                                self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                        use_attention=self.setup.use_attention))

        Tconv_W_RF_1 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[self.setup.N_u_a_strides_l2, 1],
                                       strides=(self.setup.N_u_a_strides_l2, 1),
                                       padding='valid')

        AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=round(self.setup.N_u_a / (
                                      self.setup.N_u_a_strides_l1)),
                                  n2=round(self.setup.N_b_a / (
                                          self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                  use_attention=self.setup.use_attention))

        Tconv_W_RF_2 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[self.setup.N_u_a_strides_l1, 1],
                                       strides=[self.setup.N_u_a_strides_l1, 1],
                                       padding='valid')

        current_size_W_dim_Nua = self.setup.N_u_a
        current_size_W_dim_Nba = round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2))
        for i in range(0, self.setup.n_post_Tconv_processing - 2, 1):
            AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                      kernel_size=kernels_RF,
                                      strides=self.setup.convolutional_strides,
                                      dilation=self.setup.convolutional_dilation,
                                      trainable=trainable_csi,
                                      n1=current_size_W_dim_Nua,
                                      n2=current_size_W_dim_Nba,
                                      use_attention=self.setup.use_attention))

        AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing.append(AdaSE_ResNet_class_RF(filters=1,
                                                                                          kernel_size=kernels_RF,
                                                                                          strides=self.setup.convolutional_strides,
                                                                                          dilation=self.setup.convolutional_dilation,
                                                                                          trainable=trainable_csi,
                                                                                          n1=current_size_W_dim_Nua,
                                                                                          n2=current_size_W_dim_Nba,
                                                                                          use_attention=self.setup.use_attention))

        AP_W_RF_branch_end = MaxPooling2D(pool_size=(1, current_size_W_dim_Nba))
        reshaper_W_RF = Reshape(target_shape=[self.setup.N_u_a])

        # V_RF path __________________________________________________________________________________________________
        AP_V_rf = MaxPooling3D(
            pool_size=(round(self.setup.K_prime / (self.setup.subcarrier_strides_l1 * self.setup.subcarrier_strides_l2)),
                       self.setup.Nue,
                       1))
        reshaper_V_RF_begin = Reshape(
            target_shape=[round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                          round(self.setup.N_b_a / (self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                          -1])

        AdaSE_ResNet_block_V_RF_branch = []
        # ns_adaptive_module_RF = []
        for i in range(self.setup.n_D_and_RF_layers):
            AdaSE_ResNet_block_V_RF_branch.append(AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                                                        kernel_size=kernels_RF,
                                                                        strides=self.setup.convolutional_strides,
                                                                        dilation=self.setup.convolutional_dilation,
                                                                        trainable=trainable_csi,
                                                                        n1=round(self.setup.N_u_a / (
                                                                                self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                                                        n2=round(self.setup.N_b_a / (
                                                                                self.setup.N_b_a_strides_l1 * self.setup.N_b_a_strides_l2)),
                                                                        use_attention=self.setup.use_attention))

        Tconv_V_RF_1 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[1, self.setup.N_b_a_strides_l2],
                                       strides=(1, self.setup.N_b_a_strides_l2),
                                       padding='valid')
        AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing = []
        AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=round(self.setup.N_u_a / (
                                          self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2)),
                                  n2=round(self.setup.N_b_a / (
                                      self.setup.N_b_a_strides_l1)),
                                  use_attention=self.setup.use_attention))

        Tconv_V_RF_2 = Conv2DTranspose(filters=self.setup.convolutional_filters,
                                       kernel_size=[1, self.setup.N_b_a_strides_l1],
                                       strides=[1, self.setup.N_b_a_strides_l1],
                                       padding='valid')

        current_size_V_dim_Nua = round(self.setup.N_u_a / (self.setup.N_u_a_strides_l1 * self.setup.N_u_a_strides_l2))
        current_size_V_dim_Nba = self.setup.N_b_a
        for i in range(0, self.setup.n_post_Tconv_processing - 2, 1):
            AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing.append(
                AdaSE_ResNet_class_RF(filters=self.setup.convolutional_filters,
                                      kernel_size=kernels_RF,
                                      strides=self.setup.convolutional_strides,
                                      dilation=self.setup.convolutional_dilation,
                                      trainable=trainable_csi,
                                      n1=current_size_V_dim_Nua,
                                      n2=current_size_V_dim_Nba,
                                      use_attention=self.setup.use_attention))
        AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing.append(
            AdaSE_ResNet_class_RF(filters=1,
                                  kernel_size=kernels_RF,
                                  strides=self.setup.convolutional_strides,
                                  dilation=self.setup.convolutional_dilation,
                                  trainable=trainable_csi,
                                  n1=current_size_V_dim_Nua,
                                  n2=current_size_V_dim_Nba,
                                  use_attention=self.setup.use_attention))

        AP_V_RF_branch_end = MaxPooling2D(pool_size=(current_size_V_dim_Nua, 1))
        reshaper_V_RF = Reshape(target_shape=[self.setup.N_b_a])

        # connections ------------------------------------------------------------------------------------------------

        x = AdaSE_block_common_branch[0](csi, ns)
        x = AP_1(x)
        x = AdaSE_block_common_branch[1](x, ns)
        x = AP_2(x)

        for i in range(2, self.setup.n_common_layers, 1):
            x = AdaSE_block_common_branch[i](x, ns)

        # V_RF path
        vrf = AP_V_rf(x)
        vrf = reshaper_V_RF_begin(vrf)
        vrf = AdaSE_ResNet_block_V_RF_branch[0](vrf, ns)
        for i in range(1, self.setup.n_D_and_RF_layers, 1):
            vrf = AdaSE_ResNet_block_V_RF_branch[i](vrf, ns)

        vrf = Tconv_V_RF_1(vrf)
        vrf = AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing[0](vrf, ns)
        vrf = Tconv_V_RF_2(vrf)
        for i in range(1, self.setup.n_post_Tconv_processing, 1):
            vrf = AdaSE_ResNet_block_V_RF_branch_post_Tconv_processing[i](vrf, ns)

        vrf = AP_V_RF_branch_end(vrf)
        vrf = reshaper_V_RF(vrf)

        y = reshaper_common(x)
        V_D_tmp = []
        W_D_tmp = []
        W_RF_tmp = []
        for u in range(self.setup.Nue): # implementation of parameter-binding is through this loop
            # V_D path
            vd = AdaSE_block_V_D_branch[0](y[:, :, u, :, :], ns)
            for i in range(1, self.setup.n_D_and_RF_layers, 1):
                vd = AdaSE_block_V_D_branch[i](vd, ns)

            vd = Tconv_V_D_1(vd)
            vd = AdaSE_ResNet_block_V_D_branch_post_Tconv_processing[0](vd, ns)
            vd = Tconv_V_D_2(vd)
            vd = AP_V_D_branch_end(vd)
            vd = reshaper_V_D(vd)
            for i in range(1, self.setup.n_post_Tconv_processing, 1):
                vd = AdaSE_ResNet_block_V_D_branch_post_Tconv_processing[i](vd, ns)
            vd = reshaper_V_D_end(vd)
            V_D_tmp.append(vd)

            # W_RF path
            wd = AdaSE_block_W_D_branch[0](y[:, :, u, :, :], ns)
            for i in range(1, self.setup.n_D_and_RF_layers, 1):
                wd = AdaSE_block_W_D_branch[i](wd, ns)

            wd = Tconv_W_D_1(wd)
            wd = AdaSE_ResNet_block_W_D_branch_post_Tconv_processing[0](wd, ns)
            wd = Tconv_W_D_2(wd)
            wd = AP_W_D_branch_end(wd)
            wd = reshaper_W_D(wd)
            for i in range(1, self.setup.n_post_Tconv_processing, 1):
                wd = AdaSE_ResNet_block_W_D_branch_post_Tconv_processing[i](wd, ns)
            wd = reshaper_W_D_end(wd)
            W_D_tmp.append(wd)

            # W_RF path
            wrf = AP_W_rf(y[:, :, u, :, :])
            wrf = reshaper_W_RF_begin(wrf)
            wrf = AdaSE_ResNet_block_W_RF_branch[0](wrf, ns)
            for i in range(1, self.setup.n_D_and_RF_layers, 1):
                wrf = AdaSE_ResNet_block_W_RF_branch[i](wrf, ns)

            wrf = Tconv_W_RF_1(wrf)
            wrf = AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing[0](wrf, ns)
            wrf = Tconv_W_RF_2(wrf)
            for i in range(1, self.setup.n_post_Tconv_processing, 1):
                wrf = AdaSE_ResNet_block_W_RF_branch_post_Tconv_processing[i](wrf, ns)

            wrf = AP_W_RF_branch_end(wrf)
            wrf = reshaper_W_RF(wrf)
            W_RF_tmp.append(wrf)

        V_D = tf.stack(V_D_tmp, axis=1)
        W_D = tf.stack(W_D_tmp, axis=1)
        W_RF = tf.stack(W_RF_tmp, axis=1)

        func_model = Model(inputs=[csi, ns], outputs=[V_D, vrf, W_D, W_RF], name=layer_name)
        return func_model

    @tf.function
    def custom_activation_transmitter(self, inputs):
        V_D, vrf = inputs

        V_D_cplx = tf.complex(tf.cast(V_D[:, :, :, :, :, 0], tf.float32), tf.cast(V_D[:, :, :, :, :, 1], tf.float32))
        vrf_cplx = tf.complex(tf.cast(tf.cos(vrf), tf.float32), tf.cast(tf.sin(vrf), tf.float32))

        # partially-connected analog beamformer matrix implementation ----------------

        bundeled_inputs_0 = [V_D_cplx, vrf_cplx]
        V_D_new_cplx, V_RF_cplx = tf.map_fn(self.custorm_activation_per_sample_transmitter, bundeled_inputs_0,
                                            fn_output_signature=(tf.complex64, tf.complex64),
                                            parallel_iterations=self.setup.BATCHSIZE)
        return V_D_new_cplx, V_RF_cplx

    @tf.function
    def custorm_activation_per_sample_transmitter(self, bundeled_inputs_0):
        V_D_cplx, vrf_cplx = bundeled_inputs_0
        # for BS
        vrf_zero_padded = tf.concat([tf.reshape(vrf_cplx, shape=[self.setup.N_b_a, 1]),
                                     tf.zeros(shape=[self.setup.N_b_a, self.setup.N_b_rf - 1], dtype=tf.complex64)], axis=1)
        r_bs = int(self.setup.N_b_a / self.setup.N_b_rf)
        T2_BS = []
        for i in range(self.setup.N_b_rf):
            T0_BS = vrf_zero_padded[r_bs * i: r_bs * (i + 1), :]
            T1_BS = tf.roll(T0_BS, shift=i, axis=1)
            T2_BS.append(T1_BS)
        V_RF_per_sample = tf.concat(T2_BS, axis=0)

        # repeating inputs for vectorization
        V_RF_per_sample_repeated_K_times = tf.tile([V_RF_per_sample], multiples=[self.setup.K_prime, 1, 1])

        V_D_cplx = tf.transpose(V_D_cplx, perm=[1, 0, 2, 3])  # UE, K, Nrf, Ns -> K, UE, Nrf, Ns
        bundeled_inputs_1 = [V_D_cplx, V_RF_per_sample_repeated_K_times]

        V_D_cplx_normalized_per_sample = tf.map_fn(self.normalize_power_per_subcarrier_forall_UEs_transmitter,
                                                   bundeled_inputs_1,
                                                   fn_output_signature=tf.complex64, parallel_iterations=self.setup.K_prime)
        V_D_cplx_normalized_per_sample = tf.transpose(V_D_cplx_normalized_per_sample,
                                                      perm=[1, 0, 2, 3])  # UE, K, Nrf, Ns

        return V_D_cplx_normalized_per_sample, V_RF_per_sample

    @tf.function
    def normalize_power_per_subcarrier_forall_UEs_transmitter(self, bundeled_inputs_0):
        V_D_k, V_RF = bundeled_inputs_0

        denum = 0.0
        for u in range(self.setup.Nue):
            T0 = tf.linalg.matmul(V_RF, V_D_k[u, :], adjoint_a=False, adjoint_b=False)
            T1 = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)
            denum = denum + tf.linalg.trace(T1)

        V_D_k_normalized = tf.divide(tf.multiply(V_D_k, tf.cast(tf.sqrt(self.setup.P), dtype=tf.complex64)),
                                     tf.sqrt(denum))
        return V_D_k_normalized

    @tf.function
    def custom_activation_receiver(self, inputs0):
        W_D, wrf = inputs0
        W_D_cplx = tf.complex(tf.cast(W_D[:, :, :, :, :, 0], tf.float32), tf.cast(W_D[:, :, :, :, :, 1], tf.float32))
        wrf_cplx = tf.complex(tf.cast(tf.cos(wrf), tf.float32), tf.cast(tf.sin(wrf), tf.float32))

        # partially-connected analog beamformer matrix implementation ----------------

        W_RF_cplx = tf.map_fn(self.custorm_activation_per_sample_receiver, wrf_cplx,
                              fn_output_signature=(tf.complex64),
                              parallel_iterations=self.setup.BATCHSIZE)

        return W_D_cplx, W_RF_cplx

    @tf.function
    def custorm_activation_per_sample_receiver(self, wrf_cplx):
        W_RF_cplx = tf.map_fn(self.custom_activation_per_UE, wrf_cplx,
                              fn_output_signature=(tf.complex64),
                              parallel_iterations=self.setup.Nue)
        return W_RF_cplx

    @tf.function
    def custom_activation_per_UE(self, wrf_cplx):
        # for UE
        wrf_zero_padded = tf.concat([tf.reshape(wrf_cplx, shape=[self.setup.N_u_a, 1]),
                                     tf.zeros(shape=[self.setup.N_u_a, self.setup.N_u_rf - 1], dtype=tf.complex64)], axis=1)
        r_ue = int(self.setup.N_u_a / self.setup.N_u_rf)
        T2_UE = []
        for i in range(self.setup.N_u_rf):
            T0_UE = wrf_zero_padded[r_ue * i: r_ue * (i + 1), :]
            T1_UE = tf.roll(T0_UE, shift=i, axis=1)
            T2_UE.append(T1_UE)
        W_RF_per_sample = tf.concat(T2_UE, axis=0)

        return W_RF_per_sample
