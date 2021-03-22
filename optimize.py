import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n = int(sys.argv[1])
m = n
shape = n, m


def main(n):
    cache = "set-systems/%d-%d.npy" % (n, n)
    slices_np = np.load(open(cache, "rb"))
    slices_np = slices_np.reshape((-1, n * n))
    print(slices_np.shape)
    slices = tf.constant(slices_np, dtype=tf.float32)

    def constraint(x):
        # x /= tf.reduce_sum(x)
        x = tf.clip_by_value(x, 0., 1.)
        return x
    lag = tf.Variable(np.random.uniform(size=n * n), dtype=tf.float32, constraint=constraint)
    target = tf.reduce_sum(tf.nn.relu(lag)) / tf.reduce_max(tf.tensordot(slices, lag, axes=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(-target)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(3000):
            sess.run(train_step)
            if i % 100 == 0:
                target_val, lag_val = sess.run([target, lag])
                print(i, target_val, lag_val.min(), lag_val.max(), lag_val.sum())
        best_lag = sess.run(lag).reshape((n, n))
        plt.imshow(best_lag)
        plt.show()


main(n)
