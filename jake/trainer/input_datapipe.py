import tensorflow as tf


def parse_img_example(record, target_height=128, target_width=128):
    """
    function to parse tfRecord examples back into tensors.
    """
    keys_to_features = {
        "image" : tf.FixedLenFeature((), tf.string),
        "height": tf.FixedLenFeature((), tf.int64),
        "width" : tf.FixedLenFeature((), tf.int64)
    }    
    features = tf.parse_single_example(record, keys_to_features)
    # convert features to tensors
    image = tf.decode_raw(features['image'], tf.uint8)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    # reshape input to original dimensions and cast image to type float
    image = tf.reshape(image, (height, width, 3))
    # reshape images via center crop and pad to same shape
    image = tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
    return image


def normalizer(image, dtype):
    # normalize image pixel values to within [-1,1]
    image = tf.cast(image, dtype=dtype) / 128.0 - 1.0
    # noise addition normalization
    image += tf.random_uniform(shape=tf.shape(image), minval=0., maxval=1./128., dtype=dtype)
    return image


def create_tfdataset(tfrecord_file, shuffle_buffer, epochs, batch_size, pThreads=4):
    # create a tf dataset obj from TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    # use dataset.map() in conjunction with the parse_exmp function to 
    # de-serialize each example record in TFRecord file
    dataset = dataset.map(parse_img_example, num_parallel_calls=pThreads)
    # normalize image
    dataset = dataset.map(lambda image: normalizer(image, dtype=tf.float32), num_parallel_calls=pThreads)
    # configure dataset epoch, shuffle, padding and batching operations
    dataset = dataset.shuffle(shuffle_buffer).repeat(epochs).batch(batch_size)
    return dataset