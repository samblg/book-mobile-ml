import sys
import os.path
import tensorflow as tf
import train_util as tu
import model
import numpy as np

def classify(
		image, 
		top_k, 
		k_patches, 
		ckpt_path, 
		imagenet_path):
	wnids, words = tu.load_imagenet_meta(os.path.join(imagenet_path, 'data/meta.mat'))

	image_patches = tu.read_k_patches(image, k_patches)

	x = tf.placeholder(tf.float32, [None, 224, 224, 3])

	_, pred = model.classifier(x, dropout=1.0)

	avg_prediction = tf.div(tf.reduce_sum(pred, 0), k_patches)

	scores, indexes = tf.nn.top_k(avg_prediction, k=top_k)

	saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto()) as sess:
		saver.restore(sess, os.path.join(ckpt_path, 'cnn.ckpt'))

		s, i = sess.run([scores, indexes], feed_dict={x: image_patches})
		s, i = np.squeeze(s), np.squeeze(i)

		print('AlexNet saw:')
		for idx in range(top_k):
			print ('{} - score: {}'.format(words[i[idx]], s[idx]))


if __name__ == '__main__':
	TOP_K = 5
	K_CROPS = 5
	IMAGENET_PATH = 'ILSVRC2012'
	CKPT_PATH = 'ckpt'

	image_path = sys.argv[1]

	classify(
		image_path, 
		TOP_K, 
		K_CROPS, 
		CKPT_PATH, 
		IMAGENET_PATH)


