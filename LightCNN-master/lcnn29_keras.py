def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
  x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer='he_normal', name=name)(x)
  if not use_bias:
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    bn_name = None if name is None else name + '_bn'
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  return x

def mfm(x):
  shape = K.int_shape(x)
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x1 = Cropping2D(cropping=((0, shape[3] // 2), 0))(x)
  x2 = Cropping2D(cropping=((shape[3] // 2, 0), 0))(x)
  x = Maximum()([x1, x2])
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x = Reshape([shape[1], shape[2], shape[3] // 2])(x)
  return x

def common_conv2d(net, filters, filters2, iter=1):
  res = net

  for v in range(iter):
    net = conv2d_bn(net, filters=filters, kernel_size=3, strides=1, padding='same')
    net = mfm(net)
    net = conv2d_bn(net, filters=filters, kernel_size=3, strides=1, padding='same')
    net = mfm(net)
    net = Add()([net, res]) # residual connection

  net = conv2d_bn(net, filters=filters, kernel_size=1, strides=1, padding='same')
  net = mfm(net)
  net = conv2d_bn(net, filters=filters2, kernel_size=3, strides=1, padding='same')
  net = mfm(net)

  return net

def lcnn29(inputs):
  # Conv1
  net = conv2d_bn(inputs, filters=96, kernel_size=5, strides=1, padding='same')
  net = mfm(net)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block1
  net = common_conv2d(net,filters=96, filters2=192, iter=1)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block2
  net = common_conv2d(net,filters=192, filters2=384, iter=2)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block3
  net = common_conv2d(net,filters=384, filters2=256, iter=3)

  # Block4
  net = common_conv2d(net,filters=256, filters2=256, iter=4)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  net = Flatten()(net)

  return net

input_image = Input(shape=(img_size, img_size, n_channel))

lcnn_output = lcnn29(inputs=input_image)

fc1 = Dense(512, activation=None)(lcnn_output)
fc1 = Reshape((512, 1))(fc1)
fc1_1 = Cropping1D(cropping=(0, 256))(fc1)
fc1_2 = Cropping1D(cropping=(256, 0))(fc1)
fc1 = Maximum()([fc1_1, fc1_2])
fc1 = Flatten()(fc1)

out = Dense(2, activation='linear')(fc1)
model = Model(inputs=[input_image], outputs=out)