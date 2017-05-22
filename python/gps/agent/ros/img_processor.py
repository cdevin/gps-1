import numpy as np
import rospy
from  std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

# from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
#         policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy

class ImgProcessor:
    """ 
    Listens to a camera channel and runs a network on them. Publishes the features
    to /python_image_features.
    Currently only for tf models
    """
    def __init__(self, tf_model, weights, camera_channel='/camera_crop/image_rect_color',
                 feature_channel='/tf_features_publisher'):
        """ weights should be a dictionary with variable names as keys and weights as values
        """
        print "start"
        self._camera_channel = camera_channel
        self._feature_channel = feature_channel
        self.curr_img = None
        self.fps = None
        self.sess = tf.Session()
        print "session"
        self.init_model(tf_model, weights)

        self.feature_publisher =rospy.Publisher(feature_channel, Float64MultiArray, queue_size =10)
        self.image_subscriber = rospy.Subscriber(camera_channel, Image, self._process_image)
        self.msg = None
       # self.feature_publisher =rospy.Publisher(feature_channel, Float64MultiArray, queue_size =10)
       # self.image_subscriber = rospy.Subscriber(camera_channel, Image, self._process_image)
       

    def init_model(self, tfmodel, weights):
        model = tfmodel()
        self.model = model
        op = tf.initialize_all_variables()
        self.sess.run(op)
        self.fp_tensor = model['feature_points']
        self.input_tensor = model['input']
        print "input tensro", self.input_tensor
        self.variables = model['weights']
        #import IPython
        #IPython.embed()
        if weights is not None:
            for var in self.variables:
                print var.name
                val = weights[var.name]
                assign_op = var.assign(val)
                self.sess.run(assign_op)

    def _process_image_caffe(self, msg):
        img = np.fromstring(msg.data, np.uint8)
        self.curr_img = img
        height = 240
        width = 240
        rgb_ptr = np.zeros((3*height*width,))
        for c in range(3):
            kin_c = c;
            for h in range(height):
                for w in range(width):
                    caffe_index = c * height * width + h * width + w
                    kinect_index = h*width*3 + w*3 + (2-kin_c)
                    rgb_ptr[caffe_index] = float(img[kinect_index])#// - ((float)meanrgb_[caffe_index]);
                    if c == 0:
                        rgb_ptr[caffe_index] -= 104;
                    elif c == 1:
                        rgb_ptr[caffe_index] -= 117;
                    else:
                        rgb_ptr[caffe_index] -= 123;
        
        #print img.shape
    def _process_image(self, msg):
        self.msg = msg
        img = np.fromstring(msg.data, np.uint8)
        self.curr_img =  np.reshape(img, (240,240,3))
        feed_dict = {self.input_tensor: [self.curr_img]}
        fps = self.sess.run([self.fp_tensor], feed_dict)[0][0]
        #print "fps", fps.shape
        new_msg = Float64MultiArray()
        new_msg.data = fps
        self.feature_publisher.publish(new_msg) 
        self.fps = fps
        
        
    def listen(self):

        while True:
            rospy.sleep(5)
            if self.msg is not None:
                print self.msg.header
                print self.fps







import tensorflow as tf
import numpy as np
def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))

def get_xavier_weights(filter_shape, poolsize=(2, 2), name=None):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    print filter_shape, low, high
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32), name=name)



def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def pose_prediction_network(dim_input= [240,240,3], dim_output =[4,3]):
    n_convlayers = 3
    n_layers = 3
    layer_size = 64
    pool_size = 2
    filter_size = 5
    im_height = 240; im_width = 240
    # List of indices for state (vector) data and image (tensor) data in observation.
    net_input = tf.placeholder("float", [None]+ dim_input, name='nn_input')
    poses = tf.placeholder("float", [None]+ dim_output, name='nn_output')
    # image goes through 2 convnet layers
    num_filters = [32,32, 32]
    conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
    # Store layers weight & bias
    weights = {
        'wc1': get_xavier_weights([filter_size, filter_size, 3, num_filters[0]], (pool_size, pool_size), name='wc1'), # 5x5 conv, 3 input, 32 outputs
        'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2'), # 5x5 conv, 32 inputs, 64 outputs
        'wc3': get_xavier_weights([filter_size, filter_size, num_filters[1], num_filters[2]], (pool_size, pool_size), name='wc3'),
        'wf1': init_weights([2*num_filters[2], layer_size], name='wf1'),
        #'wf1': init_weights([conv_out_size, layer_size]),
        'wf2': init_weights([layer_size, layer_size], name='wf2'),
        'wf3': init_weights([layer_size, dim_output[-1]*dim_output[-2]], name='wf3')
    }

    biases = {
        'bc1': init_bias([num_filters[0]], name='bc1'),
        'bc2': init_bias([num_filters[1]], name='bc2'),
        'bc3': init_bias([num_filters[2]], name='bc3'),
        'bf1': init_bias([layer_size], name='bf1'),
        'bf2': init_bias([layer_size], name='bf2'),
        'bf3': init_bias([dim_output[-1]*dim_output[-2]], name='bf3'),
    }

    conv_layer_0 = conv2d(img=net_input, w=weights['wc1'], b=biases['bc1'])
    #conv_layer_0 = max_pool(conv_layer_0, k=pool_size)
    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])
    #conv_layer_1 = max_pool(conv_layer_1, k=pool_size)
    #conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc3'], b=biases['bc3'])
    #==================
    #    spatial softmax
    # _, num_rows, num_cols, num_fp = conv_layer_2.get_shape()
    # print "num fp", num_fp, "num_rows", num_rows, "num_cols", num_cols
    # num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    # x_map = np.empty([num_rows, num_cols], np.float32)
    # y_map = np.empty([num_rows, num_cols], np.float32)
    # for i in range(num_rows):
    #     for j in range(num_cols):
    #         x_map[i, j] = (i - num_rows / 2.0)
    #         y_map[i, j] = (j - num_cols / 2.0)
    # x_map = tf.convert_to_tensor(x_map)
    # y_map = tf.convert_to_tensor(y_map)
    # x_map = tf.reshape(x_map, [num_rows * num_cols])
    # y_map = tf.reshape(y_map, [num_rows * num_cols])
    # # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    # features = tf.reshape(tf.transpose(conv_layer_2, [0,3,1,2]),
    #                       [-1, num_rows*num_cols])
    # softmax = tf.nn.softmax(features)
    # fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
    # fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)
    # fp = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])

    _, num_rows, num_cols, num_fp = conv_layer_1.get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols

    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)

    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])

    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layer_1, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)

    fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)

    fp = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])

    #conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])
    fc_layer1 = tf.nn.relu(tf.matmul(fp, weights['wf1']) + biases['bf1'])
    fc_layer2 = tf.nn.relu(tf.matmul(fc_layer1, weights['wf2']) + biases['bf2'])
    output = tf.reshape(tf.matmul(fc_layer2, weights['wf3']) + biases['bf3'], [-1]+dim_output)
    #loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(output-poses), reduction_indices=1, keep_dims=True)))
    loss = tf.nn.l2_loss(output[:,0,:]-poses[:,0,:])+tf.nn.l2_loss(output[:,1,:]-poses[:,1,:])+tf.nn.l2_loss(output[:,2,:]-poses[:,2,:])+tf.nn.l2_loss(output[:,3,:]-poses[:,3,:])
    #loss = tf.nn.l2_loss(output[:,3,:]-poses[:,3,:])
    #loss = tf.nn.l2_loss(output-poses)
    tensors = {'loss': loss,
               'input':net_input,
               'output': output,
               'labels': poses,
               'weights': tf.trainable_variables(),# weights,
               'biases': biases,
               'fpx': fp_x,
               'fpy': fp_y,
               'feature_points': fp,
           }
    tf.initialize_all_variables()
    return tensors



import pickle as p
print "import weights"
with open('/home/coline/code/domain_confusion/newpose_weights_5000.pkl','rb') as f:
    weights = p.load(f)
print "got weights"
ip = ImgProcessor(pose_prediction_network, weights)
rospy.init_node('image_proc',anonymous=True)
ip.listen()
