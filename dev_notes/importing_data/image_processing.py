'''
there are three common techniques for value normalization:

(x - x.min()) / (x.max() - x.min()) # values from 0 to 1
2*(x - x.min()) / (x.max() - x.min()) - 1 # values from -1 to 1
(x - x.mean()) / x.std() # values from ? to ?, but mean at 0
you’re doing pretty much the first one, without the thought that values don’t necessarily need to start at 0. thats why subtracting the min is always a good idea. the second approach is very similar, only that it’s range centers at 0.

If VGG really does it this way, they are essentially doing the first part of the third technique, meaning that the mean is zero based. Dividing by the standard deviation afterwards is always a good idea to put your values on the same scale.

as far as i know the cleanest normalization is the 3., because its the only one that centers the mean at 0, which helps a lot with exploding or disappearing gradients. that being said, i’ve never found myself in the situation where using the 3. technique instead of the 1. has given me better performance.
'''

# https://forums.fast.ai/t/images-normalization/4058/3



'''
In deep learning, there are in fact different practices as to how to subtract the mean image.

Subtract mean image
The first way is to subtract mean image as @lejlot described. But there is an issue if your dataset images are not the same size. You need to make sure all dataset images are in the same size before using this method (e.g., resize original image and crop patch of same size from original image). It is used in original ResNet paper, see reference here.

Subtract the per-channel mean
The second way is to subtract per-channel mean from the original image, which is more popular. In this way, you do not need to resize or crop the original image. You can just calculate the per-channel mean from the training set. This is used widely in deep learning, e.g, Caffe: here and here. Keras: here. PyTorch: here. (PyTorch also divide the per-channel value by standard deviation.)
https://stackoverflow.com/questions/44788133/how-does-mean-image-subtraction-work
'''

'''
http://cs231n.github.io/neural-networks-2/#datapre
'''



