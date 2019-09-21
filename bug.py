import tensorflow as tf


def main():
    with tf.Session() as sess:
        a = tf.constant(1, tf.float32)
        b = tf.constant(2, tf.float32)
        c = a + b
        sess.run([c])
    return

if __name__ == "__main__":
    main()
