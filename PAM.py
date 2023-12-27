import tensorflow as tf
import numpy as np


from FISTA import FISTA


def PAM(X, y, d):
    m = tf.shape(X)[-1]
    if len(tf.shape(X)) < 3:
        X = X[tf.newaxis, ...]
        b = 1
    else:
        b = tf.shape(X)[0]

    y = y[tf.newaxis, ...]

    A = tf.random.normal((b, d, m))
    Axt = tf.matmul(A, X, transpose_b=True)
    H = tf.matmul(Axt, y)
    V = tf.matmul(Axt, Axt, transpose_b=True)
    VH = tf.linalg.solve(V,H)
    AAt = tf.matmul(A, A, transpose_b=True)
    nu = tf.matmul(AAt,VH)
    res = FISTA(A, nu)

    return res

def PAM_piecewise(X, y, d, batch_size):
    n, m = tf.shape(X)
    num_batches = m // batch_size

    # Shuffle columns
    shuffled_indices = tf.random.shuffle(tf.range(m))
    X_shuffled = tf.gather(X, shuffled_indices, axis=1)

    # Split shuffled X into batches and stack them into a new batch dimension
    X_batches = tf.stack([X_shuffled[:, i * batch_size:(i + 1) * batch_size] for i in range(num_batches)], axis=0)

    # Apply PAM on batched X
    batch_estimation = PAM(X_batches, y, d)

    # Reshape batch_estimation and unshuffle to match original X's columns
    estimation_reshaped = tf.reshape(batch_estimation, [num_batches * batch_size, 1])
    reverse_indices = tf.argsort(shuffled_indices)
    estimation = tf.gather(estimation_reshaped, reverse_indices, axis=0)

    return estimation


if __name__ == "__main__":

    # Testing
    n = 1000
    m = 156000
    t = 10

    d = 30
    nn = 1

    X = tf.random.normal([n, m])

    num_indices = min(m, t)
    indices = np.random.choice(m, size=num_indices, replace=False)
    values = np.random.choice([-1, 1], size=num_indices)
    th = tf.zeros(m, dtype=tf.float32)
    th = tf.tensor_scatter_nd_update(th, indices[:, tf.newaxis], values)[..., tf.newaxis]

    y = tf.matmul(X, th)

    estimation = tf.zeros((1, tf.shape(X)[-1], 1))

    for _ in range(nn):
        res = PAM_piecewise(X, y, d, 1000)
        estimation += res / nn

    residual = tf.matmul(X, estimation[0, ...]) - y

    print("Solution estimation:", estimation.numpy())
    print(f"Residual : {tf.reduce_sum(residual**2)**.5}")
