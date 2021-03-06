import os
import argparse
import tensorflow as tf

import config

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--from-base', type=str, required=True, \
        help='path/to/base_model.h5')
    
    args = args.parse_args()

    return args

def ensure_args(args):
    if not os.path.exists(args.from_base):
        raise FileNotFoundError(f'{args.from_base} not found on your system!')

def load_base(path):
    base_model = tf.keras.models.load_model(path)

    return base_model

def build_siamese_net(input_shape, base_model):
    left_input = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))
    right_input = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))

    left_feature = base_model(left_input)
    right_feature = base_model(right_input)

    cosine_layer = tf.keras.layers.dot([left_feature, right_feature], axes=1, normalize=True)

    siamese_model = tf.keras.models.Model(inputs=[left_input, right_input], outputs=cosine_layer)

    return siamese_model

if __name__ == '__main__':
    args = get_args()
    ensure_args(args)

    base_model = load_base(args.from_base)
    siamese_model = build_siamese_net(config.INPUT_SHAPE, base_model)

    name = os.path.split(args.from_base)[-1]
    name = f'siamese_{name}'
    siamese_model.save(f'{config.MODEL_ROOT}/{name}')
