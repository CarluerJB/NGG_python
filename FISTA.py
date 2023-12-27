import tensorflow as tf

# Objective Function for LASSO: 0.5 * ||Ax - b||^2 + alpha * ||x||_1
def objective_function(A, x, b, alpha):
    residual = tf.matmul(A, x) - b
    l2_loss = 0.5 * tf.reduce_sum(tf.square(residual), axis=[1, 2])
    l1_loss = alpha * tf.reduce_sum(tf.abs(x), axis=[1, 2])
    return l2_loss + l1_loss, tf.math.reduce_sum(residual**2, axis=[1, 2])**.5

# Soft Thresholding Function
def soft_thresholding(x, alpha):
    return tf.sign(x) * tf.maximum(tf.abs(x) - alpha, 0)

# FISTA Update Function
def fista_update(A, b, x, y, t, alpha, lr):
    with tf.GradientTape() as tape:
        tape.watch(y)
        loss, res = objective_function(A, y, b, alpha)
    grad = tape.gradient(loss, y)

    x_new = soft_thresholding(y - lr * grad, lr * alpha)
    t_new = (1 + tf.sqrt(1 + 4 * t**2)) / 2
    y_new = x_new + ((t - 1) / t_new) * (x_new - x)

    return x_new, y_new, t_new, res

# Batched FISTA algorithm full implementation
def FISTA(A, b):
    alpha = 0.1  # Regularization parameter
    lr = 0.0001  # Learning rate
    batch_size = tf.shape(A)[0]

    # Initialize Variables
    x = tf.zeros([batch_size, tf.shape(A)[-1], 1])
    y = tf.identity(x)
    t = tf.ones([batch_size, 1, 1])

    # FISTA Iterations
    max_iterations = 1000
    for i in range(max_iterations):
        x, y, t, res = fista_update(A, b, x, y, t, alpha, lr)
        # print(f"{i} - {res*res}")
        if tf.reduce_all(res * res < 0.000001):
            break

    return y



if __name__ == "__main__":
    # Testing
    batch_size = 5
    A = tf.random.normal([batch_size, 5, 10])
    th = tf.constant([0, 0, 1, 0, 0, -1, 0, 0, 0, 0], dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]
    th = tf.repeat(th, batch_size, axis=0)

    b = tf.matmul(A, th)

    x = FISTA(A, b)

    residual = tf.matmul(A, x) - b

    print("Solution x:", x.numpy())
    print(f"Residual : {tf.reduce_sum(residual, axis=[1, 2])}")
