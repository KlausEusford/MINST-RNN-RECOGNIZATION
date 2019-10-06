import tensorflow as tf
import functools

def lazy_property(func):
    attribute='_cache_'+func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self,attribute):
            setattr(self,attribute,func(self))#if no this attribute,create the attribute with the name of funcname.
        return getattr(self,attribute)

    return wrapper


class reference(object):
    def __init__(self,
                 input_flow,labels,batch_size,time_steps,element_size):#描述输入数据的3个参数,以及输入和标签
        self.input_flow=input_flow
        self.labels=labels
        self.batch_size=batch_size
        self.time_steps=time_steps
        self.element_size=element_size

        #This section is saved some parameters of the RNN net.
        self.hidden_layer_size=128
        self.num_classes=10
        self.initial_hidden=tf.zeros([self.batch_size,self.hidden_layer_size])

        self.RNN_weights
        self.Linear_layer_weight

        #This Section is the Structures of the Net
        self.all_hidden_states
        self.all_outputs
        self.output
        #loss
        self.loss
        #train
        self.train
        #Accuracy
        self.accuracy


#=======================================================================================================================
    def rnn_step(self,previous_hidden_state,x):
        current_hidden_state=tf.tanh(
            tf.matmul(previous_hidden_state,self.RNN_weights['Wh'])+
            tf.matmul(x,self.RNN_weights['Wx'])+self.RNN_weights['Bias']
        )
        return current_hidden_state

    def get_linear_layer(self,hidden_state):
        return tf.matmul(hidden_state,self.Linear_layer_weight['Wl'])+self.Linear_layer_weight['bl']

#=======================================================================================================================

    with tf.name_scope('RNN_Weights'):
        @lazy_property
        def RNN_weights(self):
            RNN_weights=dict()
            RNN_weights['Wx']=tf.Variable(tf.zeros([self.element_size,self.hidden_layer_size]))
            RNN_weights['Wh']=tf.Variable(tf.zeros([self.hidden_layer_size,self.hidden_layer_size]))
            RNN_weights['Bias']=tf.Variable(tf.zeros([self.hidden_layer_size]))
            #记录权重
            return RNN_weights
    with tf.name_scope('Linear_layer_weights'):
        @lazy_property
        def Linear_layer_weight(self):
            Linear_layer_weight=dict()
            Linear_layer_weight['Wl']=tf.Variable(tf.truncated_normal([self.hidden_layer_size,
                                                                       self.num_classes],
                                                                      mean=0,stddev=0.01))
            Linear_layer_weight['bl']=tf.Variable(tf.truncated_normal([self.num_classes],
                                                                      mean=0,stddev=0.01))
            return Linear_layer_weight

    with tf.name_scope('all_hidden_states'):
        @lazy_property
        def all_hidden_states(self):
            return tf.scan(self.rnn_step,
                           self.input_flow,
                           initializer=self.initial_hidden,
                           name='states'
                           )

    with tf.name_scope('outputs'):
        @lazy_property
        def all_outputs(self):
            return tf.map_fn(self.get_linear_layer,self.all_hidden_states)

        @lazy_property
        def output(self):
            return self.all_outputs[-1]

    with tf.name_scope('loss'):
        @lazy_property
        def loss(self):
            cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels))
            return cross_entropy

    with tf.name_scope('Train'):
        @lazy_property
        def train(self):
            return tf.train.RMSPropOptimizer(0.001,0.9).minimize(self.loss)

    with tf.name_scope('Accuracy'):
        @lazy_property
        def accuracy(self):
            correct_prediction=tf.equal(
                                    tf.argmax(self.labels,1),
                                    tf.argmax(self.output,1))
            accuracy=(tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100
            return accuracy