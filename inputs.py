import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

class Dataset_Constant(object):
    """
    Handles training and evaluation data operations.  \n
    Data is stored as tensor constants and doesn't require initialization.
    """
    def __init__(self, images, labels, flags,
                 dtype_images=dtypes.float32, dtype_labels=dtypes.int32):
        """
        Args:
            images: string, path to 4d numpy array of images.
            labels: string, path to 2d numpy array of labels.
            flags: TensorFlow.app.Flags
            dtype_images: optional, default tensorflow.dtypes.float32
            dtype_labels: optional, default tensorflow.dtypes.int32
        """

        self.images = tf.constant(np.load(images), dtype=dtype_images)
        self.images = tf.image.crop_to_bounding_box(self.images, 5, 58, 248, 248)
        self.images = tf.image.resize_images(self.images, (128,128))
        self.labels = tf.constant(np.load(labels), dtype=dtype_labels)
        self.num_examples = self.images.shape[0]
        self.width, self.height, self.channels = self.images.shape[1:]
        assert self.images.shape[0] == self.labels.shape[0], \
            ('images.shape: %s labels.shape: %s' % (self.images.shape, self.labels.shape))
        self.flags = flags

    def _distort(self, image):
        """
        Performs distortions on an image. Currently, only illumination and noise are used.
        Args:
            image: 3D Tensor

        Returns:
            distorted_image: 3D Tensor
        """
        # Apply random noising and image flipping
        alpha = tf.random_uniform([1], maxval=0.3)
        noise = tf.random_uniform(image.shape, dtype=tf.float32)
        noised_image = (1 - alpha[0]) * image + alpha[0] * noise
        distorted_image = tf.image.random_flip_left_right(noised_image)
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        distorted_image = tf.image.per_image_standardization(distorted_image)
        return distorted_image

    def train_image_label_batch(self):
        """
        Returns: batch of training images to train on.
        """
        image_raw,label = tf.train.slice_input_producer([self.images,self.labels],
                                                      num_epochs=self.flags.num_epochs)
        image = self._distort(image_raw)
        images, labels = tf.train.batch([image,label],self.flags.batch_size,num_threads=8)

        # Display the training images in the visualizer.
        # display_images = tf.image.grayscale_to_rgb(images)
        tf.summary.image('Train_Images', images, max_outputs=6)
        return images, labels

    def eval_image_label_batch(self):
        """
        Returns: batch of evaluations images for testing.
        """
        image_raw,label = tf.train.slice_input_producer([self.images,self.labels],shuffle=False)
        image = self._distort(image_raw)
        image = tf.image.per_image_standardization(image)
        images, labels = tf.train.batch([image,label],self.flags.batch_size,num_threads=8)

        # Display the training images in the visualizer.
        tf.summary.image('Test_Images', images, max_outputs=10)
        return images, labels

class Dataset_Variable(object):
    """
    Handles training and evaluation data operations.  \n
    Data is preloaded as tensor variables and must be initialized in a session.
    """

    def __init__(self, images, labels, flags, dtype_images=dtypes.float32, dtype_labels=dtypes.int32):
        """
        Args:
            images: string, path to 4d numpy array of images.
            labels: string, path to 2d numpy array of labels.
            flags: TensorFlow.app.Flags
            dtype_images: optional, default tensorflow.dtypes.float32
            dtype_labels: optional, default tensorflow.dtypes.int32
        """

        self.images_path = images
        self.labels_path = labels
        # Read training data to get properties then remove from memory.
        images_arr = np.load(images)
        labels_arr = np.load(labels)
        self.num_examples = images_arr.shape[0]
        self.width, self.height, self.channels = images_arr.shape[1:]
        labels_shape = labels_arr.shape
        del images_arr, labels_arr

        # Create placeholders for the training data to be initialized outside of the class.
        self.images_init = tf.placeholder(
            dtype=dtype_images,
            shape=(self.num_examples,self.width,self.height,self.channels))
        self.labels_init = tf.placeholder(dtype=dtype_labels,shape=labels_shape)
        self.images = tf.Variable(self.images_init, trainable=False, collections=[])
        self.labels = tf.Variable(self.labels_init, trainable=False, collections=[])
        assert self.images.shape[0] == self.labels.shape[0], \
            ('images.shape: %s labels.shape: %s' % (self.images.shape, self.labels.shape))
        self.flags = flags

    @property
    def initializers(self):
        """

        Returns: Placeholders to initialize the training data.

        """
        return (self.images_init, self.labels_init)

    @property
    def image_data(self):
        return np.load(self.images_path)

    @property
    def label_data(self):
        return np.load(self.labels_path)

    @classmethod
    def _distort(self, image):
        random = bool(np.random.randint(0, 1))

        if random:
            distorted_image = tf.image.random_brightness(image,
                                                         max_delta=0.5)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.)
        else:
            distorted_image = tf.image.random_contrast(image,
                                                       lower=0.2, upper=1.)
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=0.5)

        image = tf.image.per_image_standardization(distorted_image)
        return distorted_image

    def train_image_label_batch(self):
        """
        Generate examples to train on.
        :param batch_size:
        :return:
        """
        image_raw,label = tf.train.slice_input_producer([self.images,self.labels],
                                                      num_epochs=self.flags.num_epochs)
        image = self._distort(image_raw)
        images, labels = tf.train.batch([image,label],self.flags.batch_size,num_threads=16)

        # Display the training images in the visualizer.
        tf.summary.image('Train_Images', images)
        return images, labels

    def eval_image_label_batch(self):
        """
        Generate examples to train on.
        :param batch_size:
        :return:
        """
        image_raw,label = tf.train.slice_input_producer([self.images,self.labels],shuffle=False)
        image = tf.image.per_image_standardization(image_raw)
        images, labels = tf.train.batch([image,label],self.flags.batch_size,num_threads=16)

        # Display the training images in the visualizer.
        tf.summary.image('Test_Images', images)
        return images, labels

class Dataset_TFRecords(object):
    """
    Handles training and evaluation data operations.  \n
    Data is read from a TFRecords filename queue.
    """
    def __init__(self, filename_queue, flags):
        self.filename_queue = filename_queue
        self.flags = flags

    def _distort(self, image, noise_min, noise_max, geometric=False):
        """
        Performs distortions on an image: noise + global affine transformations.
        Args:
            image: 3D Tensor

        Returns:
            distorted_image: 3D Tensor
        """

        # Apply random global affine transformations
        if geometric:
            # Setting bounds and generating random values for scaling and rotations
            # scale_low, scale_high = 1.0 , 1.0
            scale_X = np.random.normal(1.0, 0.02,size=1)
            scale_Y = np.random.normal(1.0, 0.02, size=1)
            # scale_X = np.random.uniform(low =scale_low, high=scale_high,size=1)
            # scale_Y = np.random.uniform(low =scale_low, high=scale_high,size=1)
            # angle_low, angle_high = np.deg2rad([-0.5, 0.5])
            theta_angle = np.random.normal(0., 0.05, size=1)
            nu_angle = np.random.normal(0.,0.05, size=1)
            # theta_angle = np.random.normal(low =angle_low, high=angle_high,size=1)
            # nu_angle = np.random.uniform(low =angle_low, high=angle_high,size=1)
            # Constructing transfomation matrix
            # scale_Y = 1.0
            a_0 = scale_X * np.cos(np.deg2rad(theta_angle))
            a_1 = -scale_Y * np.sin(np.deg2rad(theta_angle+nu_angle))
            a_2 = 0.
            b_0 = scale_X * np.sin(theta_angle)
            b_1 = scale_Y * np.cos(theta_angle+nu_angle)
            b_2 = 0.
            c_0 = 0.
            c_1 = 0.
            affine_transform = tf.constant(np.array([a_0,a_1,a_2,b_0,b_1,b_2,c_0,c_1]).flatten(),dtype=tf.float32)
            geo_image = tf.contrib.image.transform(tf.cast(image, dtype=tf.float32),affine_transform, interpolation='BILINEAR')
            geo_image = tf.image.per_image_standardization(geo_image)

        # Apply random noising and image flipping
        if geometric:
            image = geo_image

        alpha = tf.random_uniform([1], minval=noise_min, maxval=noise_max)
        image = tf.cast(image, tf.float32)
        noise = tf.random_uniform(image.shape, dtype=tf.float32)
        noised_image = (1 - alpha[0]) * image + alpha[0] * noise
        distorted_image = tf.image.per_image_standardization(noised_image)

        return distorted_image

    def _getGlimpses(self, batch_images, random=False):
        """
        Get bounded glimpses from images, corresponding to ~ 2x1 supercell
        :param batch_images: batch of training images
        :return: batch of glimpses
        """
        # set size of glimpses
        y_size, x_size = self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH
        crop_y_size, crop_x_size  = self.flags.CROP_HEIGHT,self.flags.CROP_WIDTH
        size = tf.constant(value=[crop_y_size, crop_x_size],
                           dtype=tf.int32)

        if random:
            # generate random window centers for the batch with overlap with input
            y_low, y_high = int(crop_y_size/2), int(y_size - crop_y_size/2)
            x_low, x_high = int(crop_x_size/2), int(x_size - crop_x_size/2)
            cen_y = tf.random_uniform([self.flags.batch_size], minval = y_low, maxval=y_high)
            cen_x = tf.random_uniform([self.flags.batch_size], minval=x_low, maxval=x_high)
            offsets = tf.stack([cen_y,cen_x],axis=1)
        else:
            # fixed crop
            cen_y = np.ones((self.flags.batch_size,),dtype=np.int32)*40
            cen_x = np.ones((self.flags.batch_size,), dtype=np.int32) * 70
            offsets = np.vstack([cen_y,cen_x]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)

        # extract glimpses
        glimpse_batch = tf.image.extract_glimpse(batch_images,size,offsets,centered=False,
                                                 normalized=False,
                                                 uniform_noise=False,
                                                 name='batch_glimpses')
        # print(glimpse_batch.shape)

        return glimpse_batch

    def decode_image_label(self):
        """
        Returns: image, label decoded from tfrecords
        """
        reader = tf.TFRecordReader()
        key, serialized_example  = reader.read(self.filename_queue)

        # get raw image bytes and label string
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })
        # decode from byte and reshape label and image
        label = tf.decode_raw(features['label'], tf.int64)
        label.set_shape(self.flags.NUM_CLASSES)
        image = tf.decode_raw(features['image_raw'], tf.float16)
        image.set_shape([self.flags.IMAGE_HEIGHT * self.flags.IMAGE_WIDTH * self.flags.IMAGE_DEPTH])
        image = tf.reshape(image, [self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH, self.flags.IMAGE_DEPTH])
        print(label.shape)
        return image, label

    def train_images_labels_batch(self, image_raw, label, noise_min = 0.,
                                  noise_max=0.3, random_glimpses=True, geometric=False):
        """
        Returns: batch of images and labels to train on.
        """

        # Apply image distortions
        image = self._distort(image_raw, noise_min, noise_max, geometric=geometric)

        # Generate batch
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=self.flags.batch_size,
                                                capacity=100000,
                                                num_threads=32,
                                                min_after_dequeue=10000,
                                                # shapes=image.shape,
                                                name='shuffle_batch')
        # # extract glimpses from training batch
        if random_glimpses:
            images = self._getGlimpses(images, random=True)
        else:
            images = self._getGlimpses(images)

        # Display the training images in the visualizer.
        # display_images = tf.image.grayscale_to_rgb(images)
        tf.summary.image('Train_Images', images, max_outputs=3)
        return images, labels

    def eval_images_labels_batch(self, image_raw, label, noise_min=0., noise_max=0.3,
                                 random_glimpses=False, geometric=False):
        """
        Returns: batch of images and labels to test on.
        """

        # Apply image distortions
        image = self._distort(image_raw, noise_min, noise_max, geometric=geometric)

        # Generate batch
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=self.flags.batch_size,
                                                capacity=50000,
                                                num_threads=32,
                                                min_after_dequeue=1000,
                                                name='shuffle_batch')

        #extract glimpses from evaluation batch
        if random_glimpses:
            images = self._getGlimpses(images, random=True)
        else:
            images = self._getGlimpses(images, random=False)

        # Display the training images in the visualizer.
        tf.summary.image('Test_Images', images, max_outputs=3)
        return images, labels