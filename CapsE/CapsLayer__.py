import tensorflow._api.v2.compat.v1 as tf


epsilon = 1e-9
import numpy as np

    
class CapsLayer(object):
    def __init__(self, 
                 num_outputs_secondCaps, 
                 vec_len_secondCaps, 
                 batch_size, 
                 iter_routing,
                 embedding_size, 
                 with_routing=True, 
                 layer_type='FC', 
                 sequence_length=2, 
                 useConstantInit=False,
                 filter_size=1, 
                 num_filters=50):
        
        self.num_outputs_secondCaps = num_outputs_secondCaps
        self.vec_len_secondCaps = vec_len_secondCaps
        self.with_routing = with_routing
        self.layer_type = layer_type
        self.batch_size = batch_size
        self.iter_routing = iter_routing
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.useConstantInit = useConstantInit
        self.filter_size = filter_size
        self.num_filters = num_filters


    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                if self.useConstantInit == False:
                    filter_shape = [self.sequence_length, self.filter_size, 1, self.num_filters]
                    W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W")
                else:
                    init1 = tf.constant([[[[0.1]]], [[[-0.1]]]])
                    weight_init = tf.tile(init1, [1, self.filter_size, 1, self.num_filters])
                    W = tf.compat.v1.get_variable(name="W2", initializer=weight_init)

                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv1 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                conv1 = tf.squeeze(conv1, axis=1)

                capsules =  tf.expand_dims(conv1, -1)
                capsules = squash(capsules)

                return(capsules) #[batch_size, k, num_filters, 1]

        if self.layer_type == 'FC':
            if self.with_routing:
                # Reshape the input into [batch_size, k, 1, num_filters, 1]
                self.input = tf.reshape(input, shape=(-1, input.shape[1],
                                                      1, input.shape[-2], 1))

                with tf.compat.v1.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([self.batch_size, input.shape[1], self.num_outputs_secondCaps, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ, batch_size=self.batch_size, iter_routing=self.iter_routing,
                                       num_caps_i=self.embedding_size, num_caps_j=self.num_outputs_secondCaps,
                                       len_u_i=self.num_filters, len_v_j=self.vec_len_secondCaps)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)
        
        
        
def routing(input, b_IJ, batch_size, iter_routing, num_caps_i, num_caps_j, len_u_i, len_v_j):
    # W: [num_caps_j, num_caps_i, len_u_i, len_v_j]
    W = tf.compat.v1.get_variable('Weight', shape=(1, num_caps_i, num_caps_j, len_u_i, len_v_j), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01, seed=1234))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    input = tf.tile(input, [1, 1, num_caps_j, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])

    u_hat = tf.matmul(W, input, transpose_a=True)
    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.compat.v1.variable_scope('iter_' + str(r_iter)):
            # line 4:
            c_IJ = tf.nn.softmax(b_IJ, axis=1) * num_caps_i
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)

            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                v_J = squash(s_J)
                # line 7:
                v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)

                b_IJ += u_produce_v

    return(v_J)      
    
    
def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)