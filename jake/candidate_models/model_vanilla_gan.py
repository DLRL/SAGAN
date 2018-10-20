import tensorflow as tf
tfe = tf.contrib.eager


class Generator(tf.keras.Model):
    """
    Create the generator network
    :param z: Input z
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    def __init__(self, dtype):
        super(Generator, self).__init__()

        # init all layer components; note no actual computation is done
        # fully connected layer 1
        self.fc1 = tf.keras.layers.Dense(units=4*4*512, dtype=dtype, activation='relu', name='g_fc1')
        # conv transpose + batch norm layer 1
        self.transp_conv1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, 
                                                            padding='same', activation=None, name='g_tr_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name='g_bn1')
        # conv transpose + batch norm layer 2
        self.transp_conv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2,
                                                            padding='same', activation=None, name='g_tr_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name='g_bn2')
        # conv transpose + batch norm layer 3
        self.transp_conv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2,
                                                            padding='same', activation=None, name='g_tr_conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name='g_bn3')
        # conv transpose + batch norm layer 4
        self.transp_conv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2,
                                                            padding='same', activation=None, name='g_tr_conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name='g_bn4')
        # conv transpose + batch norm layer 5
        self.transp_conv5 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=4, strides=2,
                                                            padding='same', activation=None, name='g_tr_conv5')
        self.bn5 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name='g_bn5')
        # conv2D
        self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, dtype=dtype, 
                                           padding='same', activation=None, name='g_conv')
        self.out = tf.keras.layers.Activation(activation='tanh', name='g_out')


    def call(self, z, is_training):

        net = self.fc1(z)
        net = tf.reshape(net, (-1,4,4,512), name='g_fc1_reshape')
        # first layer operation
        net = self.transp_conv1(net)
        net = self.bn1(net)
        net = tf.nn.relu(net)
        # second layer operation
        net = self.transp_conv2(net)
        net = self.bn2(net)
        net = tf.nn.relu(net)
        # third layer operation
        net = self.transp_conv3(net)
        net = self.bn3(net)
        net = tf.nn.relu(net)
        # fourth layer operation
        net = self.transp_conv4(net)
        net = self.bn4(net)
        net = tf.nn.relu(net)
        # fifth layer operation
        net = self.transp_conv5(net)
        net = self.bn5(net)
        net = tf.nn.relu(net)
        # output layer operation
        net = self.conv(net)
        output = self.out(net)

        return output


    def compute_loss(self, logits):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                              logits=logits, 
                              labels=tf.ones_like(logits)))
        return loss


class Discriminator(tf.keras.Model):    
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param alpha: Scalar value specifying the degree of leakage in leaky relu
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    def __init__(self, alpha, dtype):
        super(Discriminator, self).__init__()
        self.alpha = alpha
        # block 1
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same', 
                                            data_format='channels_last', use_bias=True, 
                                            activation=None, name='d_conv1')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', 
                                            data_format='channels_last', use_bias=True, 
                                            activation=None, name='d_conv2')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', 
                                            data_format='channels_last', use_bias=True, 
                                            activation=None, name='d_conv3')
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', 
                                            data_format='channels_last', use_bias=True, 
                                            activation=None, name='d_conv4')
        self.conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', 
                                            data_format='channels_last', use_bias=True, 
                                            activation=None, name='d_conv5')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=1, dtype=dtype, activation=None, name='d_logits')


    def call(self, inputs, is_training):
        # block 1 operation
        net = self.conv1(inputs)
        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        # block 2 operation
        net = self.conv2(net)
        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        # block 3 operation
        net = self.conv3(net)
        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        # block 4 operation
        net = self.conv4(net)
        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        # block 5 operation
        net = self.conv5(net)
        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        # logit output
        net = self.flat(net)
        logits = self.fc1(net)

        return logits
    

    def compute_loss(self, d_logits_real, d_logits_fake):
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=d_logits_real, 
                                   labels=tf.ones_like(d_logits_real)))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=d_logits_fake, 
                                   labels=tf.zeros_like(d_logits_fake)))
        return loss_real + loss_fake
