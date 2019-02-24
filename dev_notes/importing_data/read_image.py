import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# %matplotlib inline # this was originally a Jupyter Notebook
filepath = '<home directory>/imagenet/ilsvrc2012-val/n02397096/ILSVRC2012_val_00046916.JPEG'

##########################
# Approach 1: Using cv2 #
########################
# Read .JPEG image
cv2_decoded = cv2.imread(filepath)

# Rescale image and convert to float
cv2_resized = cv2.resize(cv2_decoded, (227, 227))
cv2_float = cv2_resized.astype(np.float32)

# The final image sent to the network during validation/training
cv2_img = cv2_float

# Visualization: Add mean and convert to uint8
cv2_img_viz = cv2_img.astype(np.uint8)
plt.imshow(cv2.cvtColor(cv2_img_viz, cv2.COLOR_BGR2RGB))
plt.title('Image Processed with CV2')
plt.show()


###############################
# Approach 2: Using tf.image #
#############################
# Convert filepath string to string tensor
tf_filepath = tf.convert_to_tensor(filepath, dtype=tf.string)

# Read .JPEG image
tf_img_string = tf.read_file(tf_filepath)
tf_decoded = tf.image.decode_jpeg(tf_img_string, channels=3)

# Convert to BGR
tf_bgr = tf_decoded[:, :, ::-1]

# Rescale image and convert to float
tf_resized = tf.image.resize_images(tf_bgr, [227, 227])
tf_float = tf.to_float(tf_resized)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    tf_img_bgr = sess.run(tf_bgr)
    tf_img_resized = sess.run(tf_resized_bgr)
    tf_img = sess.run(tf_float)
    
# Visualization: Add mean and convert to uint8
tf_img_viz= tf_img.astype(np.uint8)

plt.imshow(cv2.cvtColor(tf_img_viz, cv2.COLOR_BGR2RGB))
plt.title('Image Processed with TF')
plt.show()

# This cell shows where TF and CV2 operations start to diverge.
# We see these intermediate differences in two places: tf.image.decode_jpeg() and tf.image.resize_images()
# tf.image.decode_jpeg() differs from cv2.imread() in many pixels by a small amount
# tf.image.resize_images() automatically converts the dtype to float32, which may or not worsen the inaccuracy

print("tf_img_bgr (decoded + bgr): ")
print("dtype: " + str(tf_img_bgr.dtype))
print("shape: " + str(np.shape(tf_img_bgr)))
print(tf_img_bgr[:,:,0])

print("\ncv2_decoded:")
print("dtype: " + str(cv2_decoded.dtype))
print(np.shape(cv2_decoded))
print(cv2_decoded[:,:,0])


print("\n\ntf_img_resized: ")
print("dtype: " + str(tf_img_resized.dtype))
print("shape: " + str(np.shape(tf_img_resized)))
print(tf_img_resized[:,:,0])

print("\ncv2_resized:")
print("dtype: " + str(cv2_resized.dtype))
print(np.shape(cv2_resized))
print(cv2_resized[:,:,0])

print("\ncv2_float:")
print("dtype: " + str(cv2_float.dtype))
print(np.shape(cv2_float))
print(cv2_float[:,:,0])