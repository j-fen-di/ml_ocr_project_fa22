
## Proposal

### Proposal Video [Here](https://youtu.be/DIuXN7JMonY)

### Introduction
OCR, or optical character recognition, is one of the earliest addressed computer vision tasks (i.e. license scanning, CAPTCHAs, etc.). However, digitizing handwritten text specifically has seen a recent uptick in demand as technology advances. For example, people are able to cash checks from their devices using Handwriting OCR. There is also a case for accessibility, as those with vision disabilities can scan handwriting with a device and have it read to them via text conversion. Specifically, Handwriting OCR technologies are extremely popular amongst students, who often scan their handwritten notes or work for online submission.

### Problem Definition
The motivation of this project is to solve handwriting recognition with machine learning (ML) techniques, specifically relating to symbol identification and classification in regards to handwritten alphanumeric and mathematical symbols (also known as handwriting optical character recognition (OCR)). This is especially useful in applications that convert handwriting into LaTeX typesetting. Oftentimes, people must scan handwritten documents and digitize them, so high-quality handwriting recognition is vital. More specifically, college students, researchers, and professors may need to convert their handwritten work into plaintext in typesetting systems, such as LaTeX. Improving the quality of handwriting OCR can make this process more efficient.

Initially, we were planning to cover the entire scope of converting handwriting to formatted LaTeX typesetting, from OCR to combining the classified symbols in-order to create the LaTeX typesetting. However, due to our group's lack of prior knowledge in the field and the scope of the initial problem being too large, we decided to narrow the scope of the project to specifically the OCR portion in regards to mathematical symbols.

### Methods
 Due to the nature of handwriting recognition, it is important to utilize supervised learning, as the symbols identified in handwriting are often connected with their respective label or ‘bucket’. From here, classifier models will be trained to correctly place images of handwritten symbols with the correct label.

In addition, pre-processing is necessary to make the data fed to the ML algorithms usable. Some of these operations include noise removal with algorithms like Gaussian filtering method to make the image clearer, binarization to convert color and gray-scale images into a binary image, and morphological operations to make image size constant for all input images.

Finally, segmentation and feature-extraction are crucial in the handwriting recognition process. Segmentation is useful for extracting individual symbols and characters from an image. Meanwhile, feature-extraction can help an algorithm identify which characteristics or traits are associated with a character or symbol based on certain rules. For instance, a dot above a line likely means that the symbol in question is an ‘i’.

### Potential Results/Discussion
 When applying ML algorithms to handwriting recognition, a universal goal is to accurately and consistently classify handwriting as certain symbols and output the digital counterpart. To analyze how well we reach this goal, we will analyze some performance and data metrics. 

Some things to be discussed are frequency of characters, special/non-alphanumeric characters (like greek symbols), and conversion of common characters to their appropriate/equivalent functions in LaTeX. The latter introduces a unique obstacle to OCR, because context is needed. For numbers and letters, no “deep learning” is necessary. However, dots, slashes, and commas can trigger certain functions in LaTeX. Additionally, some Greek symbols can be similar shapewise to letters in the English alphabet, depending on the writer’s handwriting.

Finally, in terms of ML-related metrics, ‘accuracy’ of classification will help determine how many symbols/letters were classified with high accuracy. In addition, ‘top_k_accuracy’ is useful to analyze if symbols that recur often are classified accurately.



### Proposed Timeline
![Gannt Chart](/docs/assets/gannt.png)


## Midterm Report
### Methods
  While we initially had chosen a dataset with 100,000 image samples from Kaggle with alphanumeric symbols, Greek letters, math functions and operators included, we  realized that it was a too large a dataset to go through. It took such a long time to simply unzip every symbol folder that we thought that even preprocessing would take a very long time, so we decided to utilize a smaller dataset instead. This new dataset had numeric digits and simple math operators, and was only a size of 10,000 images, which made initial dataset manipulation a lot easier. 
  
  After getting the initial dataset of images, we first cleaned the images. Although the handwriting images looked black and white (black ink, white background) to the naked eye, when we turned the images into arrays, they turned out to be colored images, as the image arrays were 3 dimensional, not 2 dimensional. Our solution for this was to greyscale the images so that they became black and white, and therefore 2 dimensional. We also resized the images to 40 x 40 pixels so that they would all become uniform in size, and also slighly make them smaller from their initial sizes. After this, we turned the images into corresponding 2D numpy arrays, flattened them all into 1D arrays and then stacked them to make a single large array for all of our images for our input array X. 
  
  After obtaining a single input array, we then performed PCA on the array to reduce the dimensions to the first 150 components. This would serve to make building the models both easier and faster. After PCA, we performed Logistic Regression on the reduced images arraya. 

### Results and Discussion
  Initially, when we ran the Logistic Regression Model on the array with the deafult settings from sklearn, we ran into issues at the fit function. No matter what we did to the image array, we kept on getting a non convergence error. We attempted to increase the number of max iterations rate from the initial 400 all the way to 11000, but we kept on failing to get the model to converge. This meant that the model was not fitting, and we could not obtain any meaningful data from our model.

![ClassificationReport](/docs/assets/classreport.png)

Then, we tried editing the C parameter of our model, as it was previously at its default value of 1. For reference, the C parameter (also known as a hyperparameter) represents how much the model should trust the training data given. At first, a lower value of C was set for the model (i.e., C = 0.1), which meant the model did not trust the training data too much and a high amount of regularization and overfitting. This, however, came to no avail, as the model still did not converge at all. This was when we tried increasing the C parameter to 10, which meant our model trusted our training data more and a lower level of regularization and overfitting. This caused our model to fully run and finally produce data.

Upon producing the classification report, which displayed precision, recall, and F1 score metrics for each label, it was evident that the Logistic Regression Model did not have as high of a performance as expected for the math handwriting dataset. Precision ranged from 0.3 to 0.6 for the different labels, with the '0' label having the lowest precision and the '+' symbol having the highest precision. It seemed to be a trend that simpler symbols such as the '+' sign and the decimal symbol had higher precision levels in general compared to more curvy or complex symbols. Similar trends were found for recall (ability for the model to detect positive samples) and F1 score (harmonic mean of precision and recall), with the decimal symbol having the highest recall (0.94) and F1 score (0.67).

![ClassificationReportHigherC](/docs/assets/class_model_higher_c_value.png)

Finally, we increased our C parameter to a even higher value of 1,000,000, which means that our model heavily relies and trusts the given training data. Overall, there were several labels that saw a slight increase in precision, recall and F1 score, especially in labels representing symbols with a more 'complex' shape. For instance, the 'z' label saw precision increase from 0.50 to 0.54, recall from 0.38 to 0.40, and F1 score from 0.43 to 0.46. This shows that for math-based handwriting, more accurate results come from trusting real-life training data more. However, the overall results did not change much.

Some of the reasons behind the performance of the Logistic Regression Model may be due to the smaller training set given (10,000 samples vs 100,000 samples) and the Logistic Regression Model not being as a good fit for math handwriting classification compared to other models. From reading previous literature, it is hypothesized that CNN (Convolutional Neural Networks) will probably have higher performance metrics (including precision, recall, and F1 score) than our current Logistic Regression Model. We plan on utilizing CNN for our second model, so we intend to see how its results compare to our hypothesis. We also plan to do K-Cross  Validation for our Logistic Regression Model to see the mean performance metrics across different training and testing splits of our data.

### Revised Timeline with Responsibilities
![Revised Gannt Chart](/docs/assets/ganttchart_midterm_update_math_ocr.png)
