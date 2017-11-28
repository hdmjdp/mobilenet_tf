def depth_point_conv2d(X, n_ch_in, n_ch_out, kernel_size, strides, name=None,padding='SAME'):
    if name is None:
        name = 'depthconv_W'
    depth_multiplier = 1
    depthwise_shape = [kernel_size, kernel_size, n_ch_in, depth_multiplier]
    depthwise_weights = tf.get_variable(name=name + 'depthwise_weights',
                        shape=depthwise_shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.1))
    depthwise_conv = tf.nn.depthwise_conv2d(X, depthwise_weights, strides, padding)
    #num_outputs = depth_multiplier * n_ch_in
    pointwise_shape = [1, 1, n_ch_in, n_ch_out] #点卷积其实也是普通的二维卷积
    point_weights = tf.get_variable(name=name + 'point_weights',
                        shape=pointwise_shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.1))
    outputs = tf.nn.conv2d(depthwise_conv,
                     filter=point_weights,
                     strides=[1,1,1,1],
                     padding=padding)
    return outputs
