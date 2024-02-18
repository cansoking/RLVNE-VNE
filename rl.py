import tensorflow as tf
import numpy as np
import random, math

class Model():

    def __init__(self):
        self.params = {}
        self.num_sn = 100
        self.input_shape = [self.num_sn, 4, 1]
        self.num_filters_conv1 = 1
        self.x = tf.placeholder(tf.float32, shape=[None].extend(self.input_shape), name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_sn])
        self.W_conv1 = self.weight_variable([1, self.input_shape[1], self.num_filters_conv1, 1], name='Wconv', init = 'truncated_normal')
        self.b_conv1 = self.bias_variable([self.num_filters_conv1], name='bconv')
        self.h_conv1 = self.conv2d(self.x, self.W_conv1) + self.b_conv1
        self.output = tf.nn.softmax(tf.reshape(self.h_conv1, [-1, self.num_sn]))
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.output, 1e-10,1.0)))) + self.W_conv1[0][0][0][0] / 12 + self.W_conv1[0][1][0][0] / 50 
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.output, 1e-10,1.0))))
        self.sess = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        self.grad_step = self.optimizer.compute_gradients(self.loss)
        self.grads = []
        self.sess.run(tf.global_variables_initializer())

    def weight_variable(self, shape, name='unamed', init='normal'):
        if init == 'normal':
            initial = tf.random_normal(shape, stddev=0.1)
        elif init == 'truncated_normal':
            initial = tf.truncated_normal(shape, stddev=0.1)
        else:
            initial = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(initial_value=initial, name=name)

    def bias_variable(self, shape, name='unamed'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value=initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_feature_matrix(self, network, vn):
        '''
        [cpu, link_sum, degree, distance]
        '''
        feature_matrix = []
        for sn in network.substrate_nodes:
            cpu = (sn.cpu-vn.cpu) / 100
            link_sum = 0.0
            for link in sn.link_set:
                link_sum += link.bandwidth
            link_sum = link_sum / 400
            degree = len(sn.link_set)
            distance = 0.0
            num_neighbors = 0.0
            for neighbor in vn.neighbors:
                if neighbor.host is not None:
                    num_neighbors += 1
                    distance += network.floyd[sn.index][neighbor.host.index]
            if distance:
                distance = num_neighbors / distance
            feature_matrix.append([cpu, link_sum, degree, distance])
        return feature_matrix

    def choose_node(self, network, vn, req, blind=False):
        feature_matrix = np.array(self.get_feature_matrix(network, vn))
        feature_matrix = np.reshape(feature_matrix, [1, feature_matrix.shape[0], feature_matrix.shape[1], 1])
        probability = self.sess.run(self.output, feed_dict={self.x: feature_matrix})
        candidates = []
        if blind:
            rand = random.random()
            aggr = 0.0
            for index, sn in enumerate(network.substrate_nodes):
                aggr += probability[0][index]
                if aggr > rand:
                    if sn.cpu >= vn.cpu and sn not in req.hosts:
                        y = np.zeros(network.num_nodes)
                        y[index] = 1.0
                        y = np.reshape(y, [-1, network.num_nodes])
                        grad_step = self.sess.run(self.grad_step, feed_dict={self.x: feature_matrix, self.y: y})
                        self.grads.append(grad_step)
                        return sn
            return None
        else:
            prob_sum = 0.0
            for index, sn in enumerate(network.substrate_nodes):
                if sn.cpu >= vn.cpu and sn not in req.hosts:
                    candidates.append([index, probability[0][index]])
                    prob_sum += probability[0][index]
            if len(candidates) == 0.0:
                print('None')
                return None
            rand = random.random()
            if prob_sum == 0.0:
                print(probability[0])
                for candidate in candidates:
                    candidate[1] = 1.0 / len(candidates)
            else:
                rand = rand * prob_sum
            aggr = 0.0
            for candidate in candidates:
                aggr += candidate[1]
                if aggr > rand:
                    index = candidate[0]
                    y = np.zeros(network.num_nodes)
                    y[index] = 1.0
                    y = np.reshape(y, [-1, network.num_nodes])
                    grad_step = self.sess.run(self.grad_step, feed_dict={self.x: feature_matrix, self.y: y})
                    self.grads.append(grad_step)
                    return network.substrate_nodes[index]
        return None

    def update_grads(self, reward):
        tvars = tf.trainable_variables()
        gvs = []
        if len(reward) == 0:
            return
        if len(reward) == 1:
            for var_idx in range(len(tvars)):
                grad_buffer = self.grads[0][var_idx][0]
                for grad_step in self.grads[1:]:
                    grad_buffer += grad_step[var_idx][0]
                grad_buffer *= reward[0]
                gvs.append((grad_buffer, tvars[var_idx]))
        else:
            if len(reward) != len(self.grads):
                print('')
            for var_idx in range(len(tvars)):
                grad_buffer = self.grads[0][var_idx][0] * reward[0]
                for idx, grad_step in enumerate(self.grads[1:]):
                    grad_buffer += grad_step[var_idx][0] * reward[idx+1]
                gvs.append((grad_buffer, tvars[var_idx]))
        self.sess.run(self.optimizer.apply_gradients(gvs))
        self.grads = []

    def del_grad(self, counter):
        if counter:
            self.grads = self.grads[:-counter]

    def choose_node_test(self, network, vn, req, blind=False):
        feature_matrix = np.array(self.get_feature_matrix(network, vn))
        feature_matrix = np.reshape(feature_matrix, [1, feature_matrix.shape[0], feature_matrix.shape[1], 1])
        probability = self.sess.run(self.output, feed_dict={self.x: feature_matrix})
        candidates = []
        if blind:
            rand = random.random()
            aggr = 0.0
            for index, sn in enumerate(network.substrate_nodes):
                aggr += probability[0][index]
                if aggr > rand:
                    if sn.cpu >= vn.cpu and sn not in req.hosts:
                        y = np.zeros(network.num_nodes)
                        y[index] = 1.0
                        y = np.reshape(y, [-1, network.num_nodes])
                        grad_step = self.sess.run(self.grad_step, feed_dict={self.x: feature_matrix, self.y: y})
                        self.grads.append(grad_step)
                        return sn
            return None
        else:
            for index, sn in enumerate(network.substrate_nodes):
                if sn.cpu >= vn.cpu and sn not in req.hosts:
                    candidates.append([index, probability[0][index]])
            if len(candidates) == 0:
                print('None')
                return None
            candidates.sort(key=lambda c:c[1], reverse=True)
            return network.substrate_nodes[candidates[0][0]]
        return None

        
    def __del__(self):
        self.sess.close()

class ConvDense(Model):
     def __init__(self):
        self.params = {}
        self.num_sn = 100
        self.input_shape = [self.num_sn, 4, 1]
        self.num_filters_conv1 = 1
        self.x = tf.placeholder(tf.float32, shape=[None].extend(self.input_shape), name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_sn])
        self.W_conv1 = self.weight_variable([1, self.input_shape[1], self.num_filters_conv1, 1], init='truncated_normal')
        self.b_conv1 = self.bias_variable([self.num_filters_conv1])
        self.h_conv1 = tf.sigmoid(self.conv2d(self.x, self.W_conv1) + self.b_conv1)
        self.W_dense1 = self.weight_variable([100,100])
        self.b_dense1 = self.bias_variable([100])
        self.h_dense1 = tf.matmul(tf.reshape(self.h_conv1, [-1, 100]), self.W_dense1) + self.b_dense1
        self.output = tf.nn.softmax(tf.reshape(self.h_dense1, [-1, self.num_sn]))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.h_dense1))
        self.sess = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        self.grad_step = self.optimizer.compute_gradients(self.loss)
        self.grads = []
        self.sess.run(tf.global_variables_initializer())

class Dense(Model):
     def __init__(self):
        self.params = {}
        self.num_sn = 100
        self.input_shape = [self.num_sn, 4, 1]
        self.x = tf.placeholder(tf.float32, shape=[None].extend(self.input_shape), name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_sn])
        self.W_dense1 = self.weight_variable([400,100], name='Wdense1', init='normal')
        self.b_dense1 = self.bias_variable([100], name='b1')
        self.h_dense1 = tf.sigmoid(tf.matmul(tf.reshape(self.x, [1, -1]), self.W_dense1) + self.b_dense1)
        self.W_dense2 = self.weight_variable([100,100], name='Wdense2', init='normal')
        self.b_dense2 = self.bias_variable([100], name='b2')
        self.h_dense2 = tf.matmul(tf.reshape(self.h_dense1, [1, -1]), self.W_dense2) + self.b_dense2
        self.output = tf.nn.softmax(tf.reshape(self.h_dense2, [-1, self.num_sn]))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.h_dense2))
        self.sess = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        self.grad_step = self.optimizer.compute_gradients(self.loss)
        self.grads = []
        self.sess.run(tf.global_variables_initializer())
