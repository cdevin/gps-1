""" This file provides an example tensorflow network used to define a policy. """

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np


def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.batch_matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def euclidean_loss_layer(a, b, precision, batch_size):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = tf.reduce_sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor


def get_input_layer(dim_input, dim_output):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, precision


def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step))
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top


def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    return euclidean_loss_layer(a=action, b=mlp_out, precision=precision, batch_size=batch_size)


def example_tf_network(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    An example of how one might want to specify a network in tensorflow.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
    Returns:
        a TfMap object used to serialize, inputs, outputs, and loss.
    """
    n_layers = 4
    dim_hidden = (n_layers - 1) * [60]
    dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    mlp_applied = get_mlp_layers(nn_input, n_layers, dim_hidden)
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])


def multi_modal_network(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        A tfMap object that stores inputs, outputs, and scalar loss.
    """
    n_layers = 2
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden.append(dim_output)
    pool_size = 2
    filter_size = 3

    # List of indices for state (vector) data and image (tensor) data in observation.
    x_idx, img_idx, i = [], [], 0
    for sensor in network_config['obs_include']:
        dim = network_config['sensor_dims'][sensor]
        if sensor in network_config['obs_image_data']:
            img_idx = img_idx + list(range(i, i+dim))
        else:
            x_idx = x_idx + list(range(i, i+dim))
        i += dim

    nn_input, action, precision = get_input_layer(dim_input, dim_output)

    state_input = nn_input[:, 0:x_idx[-1]+1]
    image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]

    # image goes through 2 convnet layers
    num_filters = network_config['num_filters']

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(image_input, [-1, im_width, im_height, num_channels])

    # we pool twice, each time reducing the image size by a factor of 2.
    conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
    first_dense_size = conv_out_size + len(x_idx)

    # Store layers weight & bias
    weights = {
        'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size)), # 5x5 conv, 1 input, 32 outputs
        'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)), # 5x5 conv, 32 inputs, 64 outputs
    }

    biases = {
        'bc1': init_bias([num_filters[0]]),
        'bc2': init_bias([num_filters[1]]),
    }

    conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])

    conv_layer_0 = max_pool(conv_layer_0, k=pool_size)

    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

    conv_layer_1 = max_pool(conv_layer_1, k=pool_size)

    conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

    fc_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])

    fc_output = get_mlp_layers(fc_input, n_layers, dim_hidden)

    loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
    return TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss])


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def get_xavier_weights(filter_shape, poolsize=(2, 2)):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))


def multitask_multirobot_fc_supervised(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a list of dictionaries containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    #need to create taskrobot_mapping
    task_list = [2]#network_config['task_list']
    robot_list = [1]# network_config['robot_list']
    num_robots = 2
    num_tasks = 3
    tasks= range(num_tasks)
    robots=range(num_robots)
    tasks = [2]
    robots = [1]
    nnets = []
    n_layers = 6
    layer_size = 80
    dim_hidden = (n_layers - 1)*[layer_size]
    robot_weights = {}
    task_weights = {}
    dim_diff = 20
    task_out_size = 12
    keep_prob = tf.constant(1.0)
    dim_robot_specific_list = [len(range(14+18)) for r in range(num_robots)]
    dim_task_specific_list = [len(range(14,14+18+28)) for t in range(num_tasks)]
    dim_robot_output_list = [7 for r in range(num_robots)]
    # for  agent_number, agent_params in enumerate(network_config['agent_params']):
    #     print "agent", agent_number
    #     robot_number = robot_list[agent_number]; task_number = task_list[agent_number]
    #     if dim_robot_specific_list[robot_number] is not None:
    #         assert (dim_robot_specific_list[robot_number] == len(agent_params['robot_specific_idx']) and
    #                 dim_robot_output_list[robot_number] == agent_params['dim_output']), \
    #             "Robot dimentions is not consistent between agent %d and the previous ones"%agent_number
    #     if dim_task_specific_list[task_number] is not None:
    #         print dim_task_specific_list
    #         print  len(agent_params['task_specific_idx'])
    #         assert dim_task_specific_list[task_number] == len(agent_params['task_specific_idx']), \
    #             "Task dimentions is not consistent between agent %d and the previous ones"%agent_number
    #     dim_robot_specific_list[robot_number] =  len(agent_params['robot_specific_idx'])
    #     dim_robot_output_list[robot_number] = agent_params['dim_output']
    #     dim_task_specific_list[task_number] = len(agent_params['task_specific_idx'])

    for robot_number in robots:
        #special case possible
        dim_robot_output = dim_robot_output_list[robot_number]
        dim_robot_specific = dim_robot_specific_list[robot_number]
        robot_weights['w4_rn_' + str(robot_number)] = init_weights((task_out_size+ dim_robot_specific, dim_hidden[3]), name='w4_rn_' + str(robot_number))
        robot_weights['b4_rn_' + str(robot_number)] = init_bias((dim_hidden[3],), name='b4_rn_' + str(robot_number))
        robot_weights['w5_rn_' + str(robot_number)] = init_weights((dim_hidden[3], dim_hidden[4]), name='w5_rn_' + str(robot_number))
        robot_weights['b5_rn_' + str(robot_number)] = init_bias((dim_hidden[4],), name='b5_rn_' + str(robot_number))
        robot_weights['wout_rn_' + str(robot_number)] = init_weights((dim_hidden[4], dim_robot_output), name='wout_rn_' + str(robot_number))
        robot_weights['bout_rn_' + str(robot_number)] = init_bias((dim_robot_output,), name='bout_rn_' + str(robot_number))

    for task_number in tasks:
        dim_task_input = dim_task_specific_list[task_number]
        print "task", task_number
        task_weights['w1_tn_' + str(task_number)] = init_weights((dim_task_input, dim_hidden[0]), name='w1_tn_' + str(task_number))
        task_weights['b1_tn_' + str(task_number)] = init_bias((dim_hidden[0],), name='b1_tn_' + str(task_number))
        task_weights['w2_tn_' + str(task_number)] = init_weights((dim_hidden[0], dim_hidden[1]), name='w2_tn_' + str(task_number))
        task_weights['b2_tn_' + str(task_number)] = init_bias((dim_hidden[1],), name='b2_tn_' + str(task_number))
        task_weights['w3_tn_' + str(task_number)] = init_weights((dim_hidden[1], dim_hidden[2]), name='w3_tn_' + str(task_number))
        task_weights['b3_tn_' + str(task_number)] = init_bias((dim_hidden[2],), name='b3_tn_' + str(task_number))
        task_weights['taskout_tn_' + str(task_number)] = init_weights((dim_hidden[2], task_out_size), name='taskout_tn_' + str(task_number))
        task_weights['taskout_b_tn_' + str(task_number)] = init_bias((task_out_size,), name='task_out_tn_' + str(task_number))
        #task_weights['task_weights_tn_' + str(task_number)] = init_ones((task_out_size,), name='task_weights_' + str(task_number))
    tensors = {}
    tensors['keep_prob'] = keep_prob
    tensors['task_loss'] =[]
    tensors['ee_input'] =[]
    tensors['task_output'] =[]
    for agent_number, agent_params in enumerate([network_config]):
        robot_index = robot_list[agent_number]
        task_index = task_list[agent_number]

        # ee_input = tf.placeholder("float", [None, 6], name='ee_input' + str(agent_number))
        # ee_pos = ee_input[:,:3]
        # ee_vel = ee_input[:,3:]
        nn_input, action, precision = get_input_layer(dim_input, dim_output)
        robot_idx = tf.constant(range(14+18))
        task_idx = tf.constant(range(14,14+18+28))
        nn_input_t = tf.transpose(nn_input, perm=[1,0])
        robot_input = tf.transpose(tf.gather(nn_input_t, robot_idx), perm=[1,0])
        task_input = tf.transpose(tf.gather(nn_input_t, task_idx), perm=[1,0])
        print "task", task_index, "robot", robot_index
        layer1 = tf.nn.dropout(tf.nn.relu(tf.matmul(task_input, task_weights['w1_tn_' + str(task_index)]) + task_weights['b1_tn_' + str(task_index)]), keep_prob)
        layer2 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer1, task_weights['w2_tn_' + str(task_index)]) + task_weights['b2_tn_' + str(task_index)]), keep_prob)
        layer3 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer2, task_weights['w3_tn_' + str(task_index)]) + task_weights['b3_tn_' + str(task_index)]), keep_prob)
        taskout = tf.nn.relu(tf.nn.dropout(tf.matmul(layer3, task_weights['taskout_tn_'+str(task_index)]) + task_weights['taskout_b_tn_'+str(task_index)], keep_prob))
        # taskout = tf.mul(taskout, task_weights['task_weights_tn_'+str(task_index)])
        # taskout_pos = taskout[:,:3]
        # taskout_vel = taskou
        # weights = tf.sqrt(tf.reduce_sum(tf.square(ee_input[:,:3]), reduction_indices=1, keep_dims=True))
        # ee_loss =tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(ee_pos-taskout), reduction_indices=1, keep_dims=True))*weights)
        #ee_loss = tf.nn.l2_loss(ee_input-taskout)
        lastlayer_input = tf.concat(concat_dim=1, values=[taskout, robot_input])
        #lastlayer_input = tf.concat(concat_dim=1, values=[ee_input, robot_input])

        layer4 = tf.nn.dropout(tf.nn.relu(tf.matmul(lastlayer_input, robot_weights['w4_rn_' + str(robot_index)]) + robot_weights['b4_rn_' + str(robot_index)]), keep_prob)
        layer5 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer4, robot_weights['w5_rn_' + str(robot_index)]) + robot_weights['b5_rn_' + str(robot_index)]), keep_prob)
        output = tf.matmul(layer5, robot_weights['wout_rn_' + str(robot_index)]) + robot_weights['bout_rn_' + str(robot_index)]
        loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
       # tensors['task_loss'].append(ee_loss)
       # tensors['ee_input'].append(ee_input)
       #     tensors['task_output'].append(taskout)

    shared_weights = {}
    shared_weights.update(robot_weights)
    shared_weights.update(task_weights)
    return nnets[0]#, robot_weights.values(), task_weights.values(), shared_weights, tensors
