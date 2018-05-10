All implemented code for this part is using Python 3.6.

The final classifier code in the file ClassifierCNN_RawML.py.

ClassifierCNN_Segment.py is our second version of classifer, and is not as effective.

The CNN classifier requires that all test data be present as files (.png, .jpg, etc.) placed in a particular folder hierarchy.

The raw images must be placed in
./data/resources/rawTestingData/raw

where ./ is the directory where ClassifierCNN_RawML.py and all other .py files related to this project are located.

Note that the filenames must be ordered consistently via proper naming (1.png might be followed by 10.png and not 2.png, as is expected. the naming requires zero padding the string names, i.e. 0000001.png).

Nonetheless, our pretrained classifier's weights are available for download from 
https://drive.google.com/open?id=1XiRaBKQX63HH9uq0ws_7XPzTR7Ls-y7V

The weights must be placed in 
./data/models/

To run the prediction, do
python ClassifierCNN.py

Assuming all dependices are satisfied (numpy, scipy, Keras, Tensorflow), and the file hierarchy is correct, including all data dependcies, the CNN will predict!

If the testing data is not there, the program will crash with ImageGenerator flow_from_directory() exception.

If the model weights are not there, the programm will crash with Model load_weights() failed exception.

p.s. we noticed that running from command line on some systmes requires that the weights file has lowercase only filename.
