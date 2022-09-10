# Gọi các thư viện
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Bước 1 - Xử lí dữ liệu

    # Xử lí Traning set
train_datagen = ImageDataGenerator(rescale= 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')
    # Xử lí  Test set
test_datagen =  ImageDataGenerator(rescale= 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size= (64,64),
                                            batch_size= 32,
                                            class_mode= 'binary')

# Bước 2: Xây dựng model
    # Khởi tạo mạng CNN
cnn = tf.keras.models.Sequential()
    # Xây dưng lớp Convoluion đầu tiên
cnn.add(tf.keras.layers.Conv2D(32,(3,3),kernel_size=3, activation='relu', input_shape=[64,64,3]))
    # Xây dựng lớp MaxPooing
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Thêm tiếp các lớp 
cnn.add(tf.keras.layers.Conv2D(32,(3,3),kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Bước 3: Traning the CNN
cnn.compile(optimize= 'adam',loss= 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs= 25)

# Bước 4: Đưa ra dự đoán

import numpy as np
from keras_preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)




