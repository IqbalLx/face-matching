import time
import numpy as np
import cv2
import tensorflow as tf

import config

class Model:
    def __init__(self, model_path):
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, enable=True)
    
        self.model = tf.keras.models.load_model(model_path)

    def __repr__(self):
        return self.model.summary()

    @staticmethod
    def _preprocess_input(image, version=2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_temp = np.copy(image)

        image_temp = cv2.resize(image_temp, (224, 224))
        image_temp = image_temp.astype('float64')

        if version == 1:
            image_temp = image_temp[..., ::-1]
            image_temp[..., 0] -= 93.5940
            image_temp[..., 1] -= 104.7624
            image_temp[..., 2] -= 129.1863
        elif version == 2:
            image_temp = image_temp[..., ::-1]
            image_temp[..., 0] -= 91.4953
            image_temp[..., 1] -= 103.8827
            image_temp[..., 2] -= 131.0912
        else:
            raise NotImplementedError

        image_temp = np.expand_dims(image_temp, axis=0)
        
        return image_temp

    def is_match(self, first_image, second_image):
        start = time.time()

        first_image = self._preprocess_input(first_image)
        second_image = self._preprocess_input(second_image)

        score = self.model.predict([first_image, second_image])[0][0]
        score  = 1 - score

        finish = time.time() - start

        print(f"Executed in: {round(finish, 2)}s")

        return score <= config.THRESHOLD


