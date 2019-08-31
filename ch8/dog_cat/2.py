from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 128，128
input_shape = (img, width, 1img height, 3)

train_data_dir = target + 'train'
validation_data_dir = target + 'validation'

train_pic_gen = ImageDataGenerator(
      rescale=1. / 255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.5，
      horizontal_flip=True, 
      fill_mode='nearest')

validation_pic_gen = ImageDataGenerator(rescale=1, / 255)
