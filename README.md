# Handwriting-Recognition
# implementing solutions to image recognition problems
# develop programs that can read in images of hand-written digits and automatically determine which digit the image shows.
# 
# machine learning algorithm is to use a set of sample images that are already correctly labelled (called the training set) 
# and try to extract knowledge from it that it can use to later label new images (called the test set) that it has not yet seen.
# 
# The basic idea is to label a new image by finding the k most similar images in the training set and 
# assign the new image the label that is most common among those k similar images.
#
# speed up k-nearest-neighbour (kNN) by writing a parallel implementation to make use of the fact that 
# most modern computers have multiple processors over which I can spread the work.
# 
# Instead of having one process that sequentially produces the labels for all test images and then computes the accuracy, 
# a parent process will divide up the work among several child processes. 
# The parent will use pipes to tell each child which images in the test dataset it is in charge of. 
# The child will use kNN to classify the images that it was assigned by the parent and then 
# uses a pipe to report back how many of those images it classified correctly and how many it classified incorrectly. 
# The parent will compute and output the accuracy across the entire test data set based on the reports it receives from its children.
