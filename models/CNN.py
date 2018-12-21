import tensorflow as tf

# 卷积
def desnet(inputs, keep_prob_, n_classes):

    # (batch, 297，1) --> (batch, 149, 18)
    conv1 = tf.nn.relu(tf.layers.batch_normalization(inputs))
    conv1 = tf.layers.conv1d(inputs=conv1, filters=32, kernel_size=3, strides=1, padding='same')
    conv11 = tf.nn.relu(tf.layers.batch_normalization(conv1))
    conv11 = tf.layers.conv1d(inputs=conv11, filters=32, kernel_size=3, strides=1, padding='same')
    conv12 = tf.nn.relu(tf.layers.batch_normalization(conv11))
    conv12 = tf.layers.conv1d(inputs=conv12, filters=32, kernel_size=3, strides=1, padding='same')
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv12, pool_size=3, strides=2, padding='same')

    # (batch, 149, 18) --> (batch,75 , 36)
    conv2 = tf.nn.relu(tf.layers.batch_normalization(max_pool_1))
    conv2 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=3, strides=1, padding='same')
    conv21 = tf.nn.relu(tf.layers.batch_normalization(conv2))
    conv21 = tf.layers.conv1d(inputs=conv21, filters=64, kernel_size=3, strides=1, padding='same')
    conv22 = tf.nn.relu(tf.layers.batch_normalization(conv21))
    conv22 = tf.layers.conv1d(inputs=conv22, filters=64, kernel_size=3, strides=1, padding='same')
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv22, pool_size=3, strides=2, padding='same')

    # (batch, 75, 36) --> (batch, 38, 72)
    conv3 = tf.nn.relu(tf.layers.batch_normalization(max_pool_2))
    conv3 = tf.layers.conv1d(inputs=conv3, filters=128, kernel_size=3, strides=1, padding='same')
    conv31 = tf.nn.relu(tf.layers.batch_normalization(conv3))
    conv31 = tf.layers.conv1d(inputs=conv31, filters=128, kernel_size=3, strides=1, padding='same')
    conv32 = tf.nn.relu(tf.layers.batch_normalization(conv31))
    conv32 = tf.layers.conv1d(inputs=conv32, filters=128, kernel_size=3, strides=1, padding='same')
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv32, pool_size=3, strides=2, padding='same')

    # (batch, 38, 72) --> (batch, 19, 144)
    conv4 = tf.nn.relu(tf.layers.batch_normalization(max_pool_3))
    conv4 = tf.layers.conv1d(inputs=conv4, filters=256, kernel_size=3, strides=1, padding='same')
    conv41 = tf.nn.relu(tf.layers.batch_normalization(conv4))
    conv41 = tf.layers.conv1d(inputs=conv41, filters=256, kernel_size=3, strides=1, padding='same')
    conv42 = tf.nn.relu(tf.layers.batch_normalization(conv41))
    conv42 = tf.layers.conv1d(inputs=conv42, filters=256, kernel_size=3, strides=1, padding='same')
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv42, pool_size=3, strides=2, padding='same')

    # Flatten and add dropout
    flat = tf.layers.flatten(max_pool_4)
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Predictions
    logits = tf.layers.dense(flat, n_classes)
    return logits

def inference(inputs, dropout_keep_prob, n_class=2):
    return desnet(inputs, dropout_keep_prob, n_class)