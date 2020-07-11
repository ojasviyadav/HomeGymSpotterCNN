import numpy as np
import cv2
import tensorflow as tf
import matplotlib as plt
import os

cap = cv2.VideoCapture(0)

##############  BUILD THE MODEL  #############################
IMG_SIZE = 100
def buildModel():
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer_1 = tf.keras.layers.Dense(30, activation = "relu")
    prediction_layer = tf.keras.layers.Dense(3, activation = "softmax")
    model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            tf.keras.layers.Dropout(0.2),
            prediction_layer_1,
            tf.keras.layers.Dropout(0.2),
            prediction_layer,  
    ])
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])
    print('modelBuilt')
    return model

model = buildModel()
checkpoint_dir = "/Users/hetarth/Desktop/example_code/pushupDetector/models"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
model.evaluate
model.summary()
###############################################################


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    new_array = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) #reshape the frame to read
    new_array = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    new_array = tf.cast(new_array, tf.float32)
    pp = model.predict(new_array)
    pp = np.array(pp)
    np.set_printoptions(suppress=True)
    print (pp)
    org = (0,400)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    color = (255, 255, 255) 
    thickness = 2
    fontScale = 1
    frame = cv2.putText(frame, str(pp), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()