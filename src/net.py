import tensorflow as tf
import tflearn


def recon_net_large(img_inp, args):
    x = img_inp
    # Encoder
    # 64x64
    x = tflearn.layers.conv.conv_2d(
        x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x1 = x
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 32x32
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x2 = x
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 16x16
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x3 = x
    x = tflearn.layers.conv.conv_2d(
        x, 256, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 8x8
    x = tflearn.layers.conv.conv_2d(
        x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x4 = x
    x = tflearn.layers.conv.conv_2d(
        x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')

    # Decoder
    x = tflearn.layers.core.fully_connected(
        x, args.bottleneck, activation='relu', weight_decay=1e-3,
        regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, args.N_PTS*3, activation='linear', weight_decay=1e-3,
        regularizer='L2')
    x = tf.reshape(x, (-1, args.N_PTS, 3))

    return x


def recon_net_tiny_rgb_skipconn(img_inp, args):
    x = img_inp

    # Structure
    # 64 64
    x = tflearn.layers.conv.conv_2d(
        x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 32 32
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 16 16
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 8 8
    x = tflearn.layers.conv.conv_2d(
        x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 4 4
    x = tflearn.layers.core.fully_connected(
        x, args.bottleneck, activation='relu', weight_decay=1e-3,
        regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, 128, activation='relu', weight_decay=1e-3,
        regularizer='L2')
    x1 = tflearn.layers.core.fully_connected(
        x, 128, activation='relu', weight_decay=1e-3,
        regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x1, args.N_PTS*3, activation='linear', weight_decay=1e-3,
        regularizer='L2')
    x = tf.reshape(x, (-1, args.N_PTS, 3))

    # Feature
    # 64 64
    y = tflearn.layers.conv.conv_2d(
        img_inp, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 32 32
    y = tflearn.layers.conv.conv_2d(
        y, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    y = tflearn.layers.core.fully_connected(
        y, args.bottleneck, activation='relu', weight_decay=1e-3,
        regularizer='L2')
    y = tflearn.layers.core.fully_connected(
        y, 128, activation='relu', weight_decay=1e-3, regularizer='L2')
    y = tflearn.layers.core.fully_connected(
        y, 128, activation='relu', weight_decay=1e-3, regularizer='L2')
    y = tf.concat([x1, y], axis=-1)
    y = tflearn.layers.core.fully_connected(
        y, 128, activation='relu', weight_decay=1e-3, regularizer='L2')
    y = tflearn.layers.core.fully_connected(
        y, args.N_PTS*3, activation='linear', weight_decay=1e-3,
        regularizer='L2')
    y = tf.reshape(y, (-1, args.N_PTS, 3))
    y = tf.nn.sigmoid(y)

    # Structure + Feature
    z = tf.concat([x, y], axis=-1)

    return z


def recon_net_large_partseg(img_inp, args):
    x = img_inp
    # Encoder
    # 64 64
    x = tflearn.layers.conv.conv_2d(
        x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x1 = x
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 32 32
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x2 = x
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 16 16
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x = tflearn.layers.conv.conv_2d(
        x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x3 = x
    x = tflearn.layers.conv.conv_2d(
        x, 256, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    # 8 8
    x = tflearn.layers.conv.conv_2d(
        x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
        regularizer='L2')
    x4 = x
    x = tflearn.layers.conv.conv_2d(
        x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
        regularizer='L2')

    # Decoder
    x = tflearn.layers.core.fully_connected(
        x, args.bottleneck, activation='relu', weight_decay=1e-3,
        regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, 128, activation='relu', weight_decay=1e-3, regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, 128, activation='relu', weight_decay=1e-3, regularizer='L2')
    x = tflearn.layers.core.fully_connected(
        x, args.N_PTS*3, activation='linear', weight_decay=1e-3,
        regularizer='L2')
    x = tf.reshape(x, (-1, args.N_PTS, 3))

    return x
