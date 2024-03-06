# optical-character-recognition-system


    In this assignment, the goal is to implement an optical character recognition system. The first step will be a directory with an arbitrary number of target characters as individual image files. Each will represent a character to recognize in from the template. Code will be provided to read these images into an array of matrices. The second input will be a gray scale test image containing characters to recognize. 

    The OCR system will contain three parts: Enrollment, Detection and Recognition.

1, Enrollment
    In this part, code will read in characters as templates and apply scale invariant feature transform (SIFT) algorithm to extract features of character templates. These extracted features are stored as ‘descriptors’, and they will be ready to use for recognition in the third part. 
   
  

2, Detection
In this part, test image will be read in, and all characters and number in test image will be detected and draw out by blocks. In order to achieve this, my code firstly connect four neighbor components, that is finding out all connected component in an image and assigns a unique label to all points in same components. Then blocks will be draw out the extract characters and features in test image.




3, Recognition
In this part my method is to run SIFT again for test image blocks. The Sift algorithm will extract all features, therefore, the next step is to match test image features with characters template features using sum of squared differences (SSD) algorithm. Comparing with SIFT features from test images and character templates, then finding the least SSD value, which is matched characters in test image.


Finally, the matching score is 0.7209302325581395.
