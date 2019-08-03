# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import cv2
from keras.objectives import categorical_crossentropy

classes=7
train_epoch=15
batch_size=128
size=56

x=tf.placeholder(tf.float32,[None,size,size,3])
y=tf.placeholder(tf.float32,[None,classes])
train=tf.placeholder(tf.bool,[])
learning_rate=tf.placeholder(tf.float32,shape=[])
is_training=tf.placeholder(tf.bool,shape=[])
mmdloss=tf.placeholder(tf.float32,shape=[])


path_soure=r'E:\dataset\lunwenyong\qianyi\raf'
path_target=r'E:\dataset\lunwenyong\qianyi\ck'
path_test=r'E:\dataset\lunwenyong\ck'

def process(f,size,classes):
    fs = os.listdir(f)
    np.random.shuffle(fs)
    k=len(fs)
    data = np.zeros([k, size,size,3], dtype=np.float32)
    label = np.zeros([k], dtype=int)
    i = 0
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        img = cv2.imread(tmp_path)
        img = cv2.resize(img, (size,size))
        data[i]=img/img.max()
        img_label = f1[:2]
        if img_label == 'an':
            label[i] = 0
        elif img_label == 'di':
            label[i] = 1
        elif img_label == 'fe':
            label[i] = 2
        elif img_label == 'ha':
            label[i] = 3
        elif img_label == 'ne':
            label[i] = 4
        elif img_label == 'sa':
            label[i] = 5
        elif img_label == 'su':
            label[i] = 6
        else:
            print("get label error.......\n")
        #data[i][0:size*size] = np.ndarray.flatten(img)
        i = i + 1
        if i==k:
            break
    print(i)
    label = tf.keras.utils.to_categorical(label, classes)
    return data,label,i

data_0,label_0,num=process(path_soure,size,classes)
data_1,label_1,num_1=process(path_target,size,classes)
data_2,label_2,num_2=process(path_test,size,classes)
###################################################

f_of_X=tf.placeholder(tf.float32,[batch_size,2048])
f_of_Y=tf.placeholder(tf.float32,[batch_size,2048])

f_of_H=tf.placeholder(tf.float32,[batch_size,256])
f_of_Z=tf.placeholder(tf.float32,[batch_size,256])
###################################################

#线性单核
# delta = f_of_X - f_of_Y
# mmd = tf.reduce_mean(tf.matmul(tf.transpose(delta),delta))
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Guassian kernels generated from data, to be used in MK_MMD loss and/or weighted MK_MMD loss"""
    n_samples = source.get_shape()[0].value + target.get_shape()[0].value
    total = tf.concat([source, target], 0)
    total0 = tf.expand_dims(total,0)
    total0 = tf.tile(total0, multiples=(n_samples,1,1))
    total1 = tf.expand_dims(total,1)
    total1 = tf.tile(total1, multiples=(1,n_samples,1))
    L2_distance = tf.reduce_sum(tf.pow(total0 - total1, 2), 2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance)/(n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return tf.add_n(kernel_val)

def weighted_MK_MMD_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """calculate weighted MK-MMD loss.
    For normal MK-MMD loss, the target samples' probabilities in targetProb are 1"""

    batch_size = source.get_shape()[0].value
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        #w_t = tf.gather(targetProb, [s1]) * tf.gather(targetProb, [s2])
        loss += (kernels[s1, s2] + kernels[t1, t2])
        loss -= (kernels[s1, t2] + kernels[s2, t1])
    return loss / float(batch_size)
mmd=weighted_MK_MMD_loss(source=f_of_X,target=f_of_Y)
mmd2=weighted_MK_MMD_loss(source=f_of_H,target=f_of_Z)
#mokuai
def se_layer(input,pool_size,out_dim,ratio=16):
    g_pool=tf.layers.average_pooling2d(input,pool_size=pool_size,strides=pool_size,padding='same')
    excit1=tf.layers.dense(g_pool,units=out_dim//ratio)
    s_act1=tf.nn.relu(excit1)
    excit2 = tf.layers.dense(s_act1, units=out_dim)
    s_act1 = tf.nn.sigmoid(excit2)
    scale=tf.multiply(input,s_act1)
    return scale

#1111111111111111111111111111111111111111
layer_1=tf.layers.conv2d(x,filters=64,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer = tf.glorot_uniform_initializer(),
                       bias_initializer = tf.zeros_initializer())
BN_1=tf.layers.batch_normalization(layer_1,training=is_training)
act1=tf.nn.relu(BN_1)
se1=se_layer(act1,act1.shape[1].value,64)
#2222222222222222222222222222222222222222
layer_2=tf.layers.conv2d(se1,filters=64,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer = tf.glorot_uniform_initializer(),
                       bias_initializer = tf.zeros_initializer())
BN_2=tf.layers.batch_normalization(layer_2,training=is_training)
act2=tf.nn.relu(BN_2)
se2=se_layer(act2,act2.shape[1].value,64)
pool_2=tf.layers.max_pooling2d(se2,pool_size=3,strides=2,padding='same')
#3333333333333333333333333333333333333333333333333333
layer_3=tf.layers.conv2d(pool_2,filters=256,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer = tf.glorot_uniform_initializer(),
                       bias_initializer = tf.zeros_initializer())
BN_3=tf.layers.batch_normalization(layer_3,training=is_training)
act3=tf.nn.relu(BN_3)
se3=se_layer(act3,act3.shape[1].value,256)
pool3=tf.layers.max_pooling2d(se3,pool_size=3,strides=2,padding='same')
#44444444444444444444444444444444444444444444444444444444444444444444444444444444
layer_4=tf.layers.conv2d(pool3,filters=256,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer = tf.glorot_uniform_initializer(),
                       bias_initializer = tf.zeros_initializer())
bn4_2=tf.layers.batch_normalization(layer_4,training=is_training)
act4_2=tf.nn.relu(bn4_2)
se4=se_layer(act4_2,act4_2.shape[1].value,256)
#5555555555555555555555555555555555555555555555555555555555555555555555555555
layer_5=tf.layers.conv2d(se4,filters=128,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer = tf.glorot_uniform_initializer(),
                       bias_initializer = tf.zeros_initializer())
bn5_2=tf.layers.batch_normalization(layer_5,training=is_training)
act5_2=tf.nn.relu(bn5_2)
se5=se_layer(act5_2,act5_2.shape[1].value,128)
############################################################################################
layer_6=tf.layers.conv2d(se5,filters=128,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer = tf.glorot_uniform_initializer(),
                       bias_initializer = tf.zeros_initializer())
bn6_2=tf.layers.batch_normalization(layer_6,training=is_training)
act6_2=tf.nn.relu(bn6_2)
se6=se_layer(act6_2,act6_2.shape[1].value,128)
pool6=tf.layers.max_pooling2d(se6,pool_size=3,strides=2,padding='same')

fc=tf.layers.flatten(pool6)
fc1=tf.layers.dense(fc,2048)
bn_fc1=tf.layers.batch_normalization(fc1,training=is_training)
act_fc1=tf.nn.relu(bn_fc1)
fc2=tf.layers.dense(act_fc1,256)
bn_fc2=tf.layers.batch_normalization(fc2,training=is_training)
act_fc2=tf.nn.relu(bn_fc2)

output=tf.layers.dense(act_fc2,classes,activation=tf.nn.softmax)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
cost = tf.reduce_mean(categorical_crossentropy(y,output))
######################################
total=cost+0.6*mmdloss

#####################################
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(total)

correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accurary',accuracy)

merged=tf.summary.merge_all()


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer=tf.summary.FileWriter(r"E:\tensorboard\ck",sess.graph)
    init_rate=0.01
    best_acc=0
    for epoch in range(0, train_epoch):
        init_rate=init_rate/(1+0.0005*epoch)
        for i in range(0, num, batch_size):
            if num-i<batch_size:
                i=num-batch_size
            else:
                pass
            data=data_0[i:i+batch_size,:,:,:]
            label=label_0[i:i+batch_size,:]
            #
            loss1 ,loss3= sess.run([fc1,fc2], feed_dict={x:data ,is_training: False})
            loss2 ,loss4= sess.run([fc1,fc2], feed_dict={x:data_1[i:i + batch_size, :, :, :], is_training: False})
            aaa = sess.run(mmd, feed_dict={f_of_X: loss1, f_of_Y:loss2})
            bbb=sess.run(mmd2, feed_dict={f_of_H: loss3, f_of_Z:loss4})
            #print(aaa)
            sess.run(optimizer,feed_dict={x: data, y: label, learning_rate: init_rate,
                                          is_training: True, mmdloss:(aaa+bbb)})

        test_loss, test_acc, summary = sess.run([cost, accuracy, merged], feed_dict={x: data_2, y: label_2, is_training:False})
        if test_acc>best_acc:
            best_acc=test_acc
        writer.add_summary(summary, epoch)
        print("Epoch: " + str(epoch + 1) + ", Test Loss= " + "{:.3f}".format(test_loss) + ", Test Accuracy= " + "{:.3f}".format(test_acc))
    print('best_acc=',best_acc)
print('ok')
