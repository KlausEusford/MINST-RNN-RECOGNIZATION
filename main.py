
import Reference
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE=128
TIME_STEPS=28
ELEMENT_SIZE=28
LOG_DIR='logs/RNN_with_summaries'

mnist=input_data.read_data_sets(r'F:\tensorflow_basic_usage_oop\input_data',one_hot=True)

#定义两个占位符，分别是原始输入&labels
placeholder_raw_input=tf.placeholder(tf.float32,shape=[None,TIME_STEPS,ELEMENT_SIZE],name='inputs')
placeholder_labels=tf.placeholder(tf.float32,shape=[None,10],name='Labels')

processed_input=tf.transpose(placeholder_raw_input,perm=[1,0,2])

Simple_RNN=Reference.reference(processed_input,placeholder_labels,BATCH_SIZE,TIME_STEPS,ELEMENT_SIZE)#定义了一个新的网络对象

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):#10000--1000
        batch_x,batch_y=mnist.train.next_batch(BATCH_SIZE)#读取下一个batch的值b
        batch_x=batch_x.reshape([BATCH_SIZE,TIME_STEPS,ELEMENT_SIZE])
        sess.run(Simple_RNN.train,feed_dict={placeholder_raw_input:batch_x,placeholder_labels:batch_y})

        if i % 1000==0:#1000---100
            acc,loss=sess.run([Simple_RNN.accuracy,Simple_RNN.loss],feed_dict={placeholder_raw_input:batch_x,placeholder_labels:batch_y})
            print("Iter"+str(i)+",Mnibatch Loss="+"{:.6f}".format(loss)+",Training Accuracy="+"{:.5f}".format(acc))


    for i in range(10):
        test_data,test_labels=mnist.test.next_batch(BATCH_SIZE)
        test_data=test_data.reshape([BATCH_SIZE,TIME_STEPS,ELEMENT_SIZE])
        test_acc=sess.run(Simple_RNN.accuracy,feed_dict={placeholder_raw_input:test_data,placeholder_labels:test_labels})
        print("Iter"+str(i)+",Testing Accuracy="+"{:.5f}".format(test_acc))
