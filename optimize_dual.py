import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

n = int(sys.argv[1])
m = n
shape = n, m

do_sparse = True

def llp_to_sparsetensor(llp):
    indices = []
    values = []
    for i in range(len(llp)):
        for x, y in llp[i]:
            indices.append((i, x * n + y))
            values.append(1)
    return tf.sparse.SparseTensor(np.array(indices, dtype=np.int64), np.array(values, dtype=np.float32), dense_shape=(len(llp), n * n))


def main(n):
    if do_sparse:
        # list of list of pairs
        cache = pickle.load(open("dual-set-systems/%d-%d.pkl" % (n, n), "rb"))
        slices = llp_to_sparsetensor(cache)
    else:
        cache = "dual-set-systems/%d-%d.npy" % (n, n)
        slices_np = np.load(open(cache, "rb"))
        slices_np = slices_np.reshape((-1, n * n))
        print(slices_np.shape)
        slices = tf.sparse.from_dense(tf.constant(slices_np, dtype=tf.float32))

    def constraint(x):
        # x /= tf.reduce_sum(x)
        x = tf.clip_by_value(x, 0., 1.)
        return x

    d = 2
    coeffs = tf.concat([tf.Variable(np.ones(1), dtype=tf.float32), tf.Variable(np.ones((d+1)*(d+1) - 1), dtype=tf.float32)], axis=0)

    parametric = False
    if parametric:
        # this codepath does not yet support float16
        xvar = tf.expand_dims(tf.linspace(-1., 1., n), 0)
        yvar = tf.expand_dims(tf.linspace(-1., 1., n), 1)

        # these are our building blocks:
        X1 = tf.abs(xvar)
        X2 = xvar * xvar
        Y1 = tf.abs(yvar)
        Y2 = yvar * yvar
        D1 = X1 + Y1
        D2 = X2 + Y2
        Dinf = tf.maximum(X1, Y1)

        '''
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6], coeffs[7], coeffs[8], coeffs[9]

        V = D1
        # for n=20: poly(Dinf) ~ 13.01, poly(D1) ~ 13.23, poly(D2) ~ 13.43.
        lag = c0 + c1 * V + c2 * V * V + c3 * V * V * V + c4 * V * V * V * V

        lag = c0 + c1 * X2 + c2 * Y2 + c3 * X2 * Y2

        # lag = c0 + c1 * R2 + c2 * R2 * R2 + c3 * R2 * R2 * R2 + c4 * R2 * R2 * R2 * R2
        lag = c0 + c1 * X2 + c2 * Y2 + c3 * X2 * Y2 + c4 * X2 * X2 + c5 * Y2 * Y2 + c6 * X2 * X2 * Y2 + c7 * X2 * Y2 * Y2 + c8 * X2 * X2 * Y2 * Y2
        # lag = c0 + c1 * X2 + c1 * Y2 + c3 * X2 * Y2
        # lag = c0 * 0 + X2 + Y2 - 2 * X2 * Y2

        lag = c0 + c1 * X1 + c2 * Y1 + c3 * X1 * Y1 + c4 * X1 * X1 + c5 * Y1 * Y1 + c6 * X1 * X1 * Y1 + c7 * X1 * Y1 * Y1 + c8 * X1 * X1 * Y1 * Y1
        '''

        Y11 = tf.abs(Y1 - 0.5)
        lag = 0
        for i in range(d + 1):
            for j in range(d + 1):
                lag += coeffs[i * (d+1) + j] * X1 ** i * Y11 ** j

        # lag = c0 + c1 * X1 + c2 * Y1

        # lag = c0 + c1 * Dinf + c2 * tf.sqrt(D2) + c3 * D1 + c4 * X2 + c5 * Y2 + c6 * X2 * Y2

        lag = tf.reshape(lag, [-1])
    else:
        # this is the nonparametric solution:
        lag = tf.Variable(np.ones(n * n), dtype=tf.float32, constraint=constraint)

    target = tf.reduce_sum(tf.nn.relu(lag)) / tf.reduce_min(tf.sparse_tensor_dense_matmul(slices, tf.reshape(lag, (n*n, 1))))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
        global_step, 1000, 0.8, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(target, global_step=global_step)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(50000):
            sess.run(train_step)
            if i % 100 == 0:
                target_val, lag_val = sess.run([target, lag])
                print(i, target_val, lag_val.min(), lag_val.max(), lag_val.sum())
                np.save(open("dual-tensorflow.%d-%d.temp.npy" % shape, "wb"), lag_val.reshape((n, n)))
        best_lag = sess.run(lag).reshape((n, n))
        np.save(open("dual-tensorflow.%d-%d.npy" % shape, "wb"), best_lag)
        best_coeffs = sess.run(coeffs)
        print("best_coeffs", best_coeffs)
        plt.imshow(best_lag.T)
        plt.show()


main(n)
