#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:30:29 2017

@author: Chris

test of bit to int methodic in tensorflow
"""

import tensorflow as tf
import numpy as np

binary_string = tf.constant([1, 0, 0, 1, 1, 0,0,0,0,1,0,0,0,1,1], shape=[5,5], dtype=tf.int64)

result2 = tf.map_fn(lambda x: tf.reduce_sum(
    tf.cast(tf.reverse(tensor=x, axis=[0]), dtype=tf.int64)
    * 2 ** tf.range(tf.cast(5, dtype=tf.int64))), binary_string)

a = tf.constant([1, 0, 0, 1, 1])
b = tf.constant([1, 0, 0, 1, 1])
a = tf.reshape(a, shape=[1, 5])
b = tf.reshape(b, shape=[1, 5])

c = tf.concat((a, b), axis = 0)



""" for only one element:
result = tf.reduce_sum(
    tf.cast(tf.reverse(tensor=binary_string, axis=[0]), dtype=tf.int64)
    * 2 ** tf.range(tf.cast(5, dtype=tf.int64)))
"""
    
with tf.Session():

    
#    print(multiDim_range.eval())
    
    print(result2.eval()[3])

    print(c.eval())
#%%
zero = tf.constant(1)
one = tf.constant(1)
zero2 = tf.constant([0])
one2 = tf.constant([1])
def f1(): return tf.constant(1)
def f2(): return tf.constant(0)
 
binary_string = tf.constant([0, 1])

result = tf.argmax(binary_string)

result2 = tf.cond(tf.equal(zero, binary_string[1]) & tf.equal(zero, zero), f1, f2)

"""
result2 = tf.map_fn(lambda x: tf.reduce_sum(
    tf.cast(tf.reverse(tensor=x, axis=[0]), dtype=tf.int64)
    * 2 ** tf.range(tf.cast(5, dtype=tf.int64))), binary_string)

"""
    
with tf.Session():

        
    print(result.eval())
    print(result2.eval())

#%%
import tensorflow as tf

tf.reset_default_graph()

binary_string = tf.constant([1, 0, 0, 1, 1, 0,0,0,0,1,0,0,0,1,1], shape=[3,5], dtype=tf.int64)

result2 = tf.Variable((0,0,0))

i = tf.constant(0)
while_contition = lambda i : tf.less(i, tf.constant(8, tf.int32))
def body(i):
    
    # aus 5 bit in einem Vector eine Zahl machen
    result2[i] = tf.map_fn(lambda x: tf.reduce_sum(
        tf.cast(tf.reverse(tensor=x, axis=[0]), dtype=tf.int64)
        * 2 ** tf.range(tf.cast(5, dtype=tf.int64))), binary_string)
  
    return [tf.add(i,1)]

r = tf.while_loop(while_contition, body, [i])

    
with tf.Session():
    r.eval()
    print(result2.eval())

#%%

