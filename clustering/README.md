# Computer vision with Clustering

http://www.cs.unc.edu/~lazebnik/spring09/lec18_bag_of_features.pdf



bag of features

영상의 각 부분에서 Feature를 추출하여 일종의 histogram을 만듦으로써 이미지를 분석하는 기법. Bag of word에서 유래하였다.



1. Extract features ( Detect patches -> normalize patch -> compute descriptor)
2. Learn visual vocabulary -> clustering (k-means)
3. Quantize feature with visual vocabulary
4. Represent images by frequencies of visual words