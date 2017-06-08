# DeepLearn_STEM
Deep Convolutional Neural Networks for infering Complex Oxide Octahedral Tilts from ABF STEM data.  
The python scripts setup a CNN, perform training, evaluation, and save checkpoints and tensorbard summary data for visualization.  
Code requires a GPU-enabled installation of TensorFlow v1.1 and higher.  
Here's a brief description:  
1. __inputs.py__:  
  Helper classes to read training/evaluation data, create training batches, and image distortion.  
  Can handle I/O ops on TFRecords and numpy arrays.
2. __network.py__:  
  Helper functions to setup the CNNs.  
3. __training.py__:  
  Python executable to start network training, also sets FLAGS that describe data, saving, training params.  
4. __evaluation.py__:  
  Python executable to evaluate network precision, etc... 
  


 


