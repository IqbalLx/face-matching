import argparse
from keras_vggface.vggface import VGGFace

import config

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-m', '--model', required=True, type=str,
                    help="Choose between vgg16, resnet50, or senet50")
    args.add_argument('-s', '--shape', type=int, default=224,
                    help="Provide input dimensions. default to 224")
    args.add_argument('-p', '--pooling', type=str, default='max',
                    help="Select last pooling layer between max or avg. default to max")
    args = args.parse_args()

    return args

def ensure_args(args):
    assert args.model in ["vgg16", "resnet50", "senet50"]
    assert args.pooling in ["max", "avg"]

if __name__ == "__main__":
    args = get_args()
    ensure_args(args)
    
    base_model = VGGFace(
        model=args.model, 
        include_top=False, 
        input_shape=(args.shape, args.shape, 3), 
        pooling=args.pooling
        )

    model_name = f"{config.MODEL_ROOT}/{base_model.name}_{args.shape}.h5"
    base_model.save(model_name)
    
