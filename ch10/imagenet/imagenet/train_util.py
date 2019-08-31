import os
import random
import tensorflow as tf 
import numpy as np
from scipy.io import loadmat
from PIL import Image


def weight(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.01)
	w = tf.Variable(initial, name=name)
	tf.add_to_collection('weights', w)
	return w

def bias(value, shape, name):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, stride, padding):
	return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool2d(x, kernel, stride, padding):
	return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padding)

def lrn(x, depth_radius, bias, alpha, beta):
	return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)

def relu(x):
	return tf.nn.relu(x)

def batch_norm(x):
	epsilon = 1e-3
	batch_mean, batch_var = tf.nn.moments(x, [0])
	return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)

def onehot(index):
	onehot = np.zeros(1000)
	onehot[index] = 1.0
	return onehot

def read_batch(batch_size, images_source, wnid_labels):
	batch_images = []
	batch_labels = []

	for i in range(batch_size):
		class_index = random.randint(0, 999)

		folder = wnid_labels[class_index]
		batch_images.append(read_image(os.path.join(images_source, folder)))
		batch_labels.append(onehot(class_index))

	np.vstack(batch_images)
	np.vstack(batch_labels)
	return batch_images, batch_labels

def read_image(images_folder):
	image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder)))
	
	im_array = preprocess_image(image_path)
		
	return im_array

def preprocess_image(image_path):
	IMAGENET_MEAN = [123.68, 116.779, 103.939] 

	img = Image.open(image_path).convert('RGB')

	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_cropped = img.crop((x, y, x + 224, y + 224))

	cropped_im_array = np.array(img_cropped, dtype=np.float32)

	for i in range(3):
		cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

	return cropped_im_array


def read_k_patches(image_path, k):
	IMAGENET_MEAN = [123.68, 116.779, 103.939] 

	img = Image.open(image_path).convert('RGB')

	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	patches = []
	for i in range(k):
		x = random.randint(0, img.size[0] - 224)
		y = random.randint(0, img.size[1] - 224)
		img_cropped = img.crop((x, y, x + 224, y + 224))

		cropped_im_array = np.array(img_cropped, dtype=np.float32)

		for i in range(3):
			cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

		patches.append(cropped_im_array)

	np.vstack(patches)
	return patches

def read_validation_batch(batch_size, validation_source, annotations):
	batch_images_val = []
	batch_labels_val = []

	images_val = sorted(os.listdir(validation_source))

	with open(annotations) as f:
		gt_idxs = f.readlines()
		gt_idxs = [(int(x.strip()) - 1) for x in gt_idxs]

	for i in range(batch_size):
		idx = random.randint(0, len(images_val) - 1)

		image = images_val[idx]
		batch_images_val.append(preprocess_image(os.path.join(validation_source, image)))
		batch_labels_val.append(onehot(gt_idxs[idx]))

	np.vstack(batch_images_val)
	np.vstack(batch_labels_val)
	return batch_images_val, batch_labels_val

def load_imagenet_meta(meta_path):
	metadata = loadmat(meta_path, struct_as_record=False)
	
	synsets = np.squeeze(metadata['synsets'])
	ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
	wnids = np.squeeze(np.array([s.WNID for s in synsets]))
	words = np.squeeze(np.array([s.words for s in synsets]))
	return wnids, words

def read_test_labels(annotations_path):
	gt_labels = []

	with open(annotations_path) as f:
		gt_idxs = f.readlines()
		gt_idxs = [(int(x.strip()) - 1) for x in gt_idxs]

	for gt in gt_idxs:
		gt_labels.append(onehot(gt))

	np.vstack(gt_labels)

	return gt_labels

def format_time(time):
	m, s = divmod(time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))

def imagenet_size(im_source):
	n = 0
	for d in os.listdir(im_source):
		for f in os.listdir(os.path.join(im_source, d)):
			n += 1
	return n
