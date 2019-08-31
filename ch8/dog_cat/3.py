train_flow= train_pic_gen.flow_from_directory(
      train_data_dir,
      target_size=(img_width, img_height),
      batch_size=32，
      class_mode='binary'
)

validation_flow = validation_pic_gen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      batch_size=32，
      class_mode='binary'
)

