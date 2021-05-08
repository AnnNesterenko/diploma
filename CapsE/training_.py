import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from CapsE__ import *


def train_step(x_batch, y_batch, capse, sess, train_op, global_step):
    feed_dict = {capse.input_x: x_batch, capse.input_y: y_batch}
    _, step, loss = sess.run([train_op, global_step, capse.total_loss], feed_dict)
    return loss


def predict(x_batch, y_batch, capse, sess):
    feed_dict = {capse.input_x: x_batch, capse.input_y: y_batch}
    scores = sess.run([capse.predictions], feed_dict)
    return scores


def test_prediction(x_batch, y_batch, lstOriginalRank, capse, sess, dataset='ARXIV'):
    new_x_batch = np.concatenate(x_batch)
    new_y_batch = np.concatenate(y_batch, axis=0)

    while len(new_x_batch) % (BATCH_SIZE * 20) != 0:
        new_x_batch = np.append(new_x_batch, np.array([new_x_batch[-1]]), axis=0)
        new_y_batch = np.append(new_y_batch, np.array([new_y_batch[-1]]), axis=0)
    
    results = []
    listIndexes = range(0, len(new_x_batch), 20 * BATCH_SIZE)
    
    for tmpIndex in range(len(listIndexes) - 1):
        results = np.append(results,
                            predict(new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                                    new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                                    capse, sess))
    results = np.append(results,
                        predict(new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:], capse, sess))

    if dataset == 'MIND':
        return results 

    lstresults = []
    _start = 0
    for tmp in lstOriginalRank:
        _end = _start + len(tmp)
        lstsorted = np.argsort(results[_start:_end])
        lstresults.append(np.where(lstsorted == 0)[0] + 1)
        _start = _end    

    return lstresults


def computeP1(lstRanks):
    p1 = 0.0
    for tmp in lstRanks:
        if tmp[0] == 1:
            p1 += 1
    return p1 / len(lstRanks)


def computeM1(results, lstOriginalRank):
    m = 0.0
    _start = 0
    for tmp in lstOriginalRank:
        _end = _start + len(tmp)
        s = np.sum(tmp) # сколько единиц в начале
        lstsorted = np.argsort(results[_start:_end])[:s]
        t = len(set(lstsorted) & set(np.arange(s))) == s
        m += int(t)
        _start = _end 
    return m / len(lstOriginalRank)


def computeM2(results, lstOriginalRank):
    m = 0.0
    _start = 0
    for tmp in lstOriginalRank:
        _end = _start + len(tmp)
        s = np.sum(tmp) # сколько единиц в начале
        lstsorted = np.argsort(results[_start:_end])[:s]
        t = len(set(lstsorted) & set(np.arange(s))) / s
        m += t
        _start = _end 
    return m / len(lstOriginalRank)


def train_model(lst_embeddings_query,
                lst_embeddings_doc,
                data_size,
                train_batch,
                test_duplets,
                test_val_duplets,
                num_filters=50, 
                epochs=100, 
                learning_rate=0.00001,
                dataset='ARXIV'):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            global_step = tf.Variable(0, name="global_step", trainable=False)

            capse = CapsE(sequence_length=2,
                          batch_size=20 * BATCH_SIZE,
                          initialization=[lst_embeddings_query, lst_embeddings_doc],
                          embedding_size=200,
                          filter_size=1,
                          num_filters=num_filters,
                          iter_routing=1,
                          num_outputs_secondCaps=1,
                          vec_len_secondCaps=2)

            # Define Training procedure
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(capse.total_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            lsttest = []
            losses = []

            num_batches_per_epoch = int((data_size - 1) / (BATCH_SIZE)) + 1
            
            for epoch in range(1, epochs + 1):
                for batch_num in range(num_batches_per_epoch):
                    x_batch, y_batch = train_batch()
                    loss = train_step(x_batch, y_batch, capse, sess, train_op, global_step)
                    current_step = tf.train.global_step(sess, global_step)
                    
                losses.append(loss)

                if dataset == 'ARXIV':
                    test_results = test_prediction(test_duplets, test_val_duplets, test_val_duplets, capse, sess)
                    test_p1 = computeP1(test_results)
                    lsttest.append(test_p1)
                    
                
                if dataset == 'MIND':
                    test_results = test_prediction(test_duplets, test_val_duplets, test_val_duplets, capse, sess, dataset='MIND')
                    test_m1 = computeM1(test_results, test_val_duplets)
                    test_m2 = computeM2(test_results, test_val_duplets)
                    lsttest.append([test_m1, test_m2])
                
                
    return lsttest, losses