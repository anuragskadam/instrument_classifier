# Instrument Classifier  
In this project, I have implemented a CNN based architecture to classify a 5 second audio (.wav file with sampling rate 44.1 kHz) clip into three musical instruments (Guitar, Drums, Piano) based on  which instrument is being played in the clip.  
  
The model takes as input the spectrogram of the audio signal, and outputs the pre-softmaxed probabilities for each instrument respectively.  
  
The dataset was downloaded from https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset/data.  
However the dataset was very poorly labeled and I and my friend had to re-label about 1300 data-points ourselves.  

Since the model and dataset were too large to be uploaded to Github, I have only uploaded the main Jupyter Notebook that does both the training and testing for the model.  
I have also uploaded the Notebook I used for dataprocessing (not clean).

