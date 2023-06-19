# Handwriting OCR for Mathematics - Final Report
## Team Members 
Justin Effendi, Deborah Fidler, Trinity Davies, Rudresh Patel


## Final Report Video [Here](https://youtu.be/ZqAUeD1Efto)

## Introduction
OCR, or optical character recognition, is one of the earliest addressed computer vision tasks (i.e. license scanning, CAPTCHAs, etc.). However, digitizing handwritten text specifically has seen a recent uptick in demand as technology advances. For example, people are able to cash checks from their devices using Handwriting OCR. There is also a case for accessibility, as those with vision disabilities can scan handwriting with a device and have it read to them via text conversion. Specifically, Handwriting OCR technologies are extremely popular amongst students, who often scan their handwritten notes or work for online submission.

## Problem Definition
The motivation of this project is to solve handwriting recognition with machine learning (ML) techniques, specifically relating to symbol identification and classification in regards to handwritten alphanumeric and mathematical symbols (also known as handwriting optical character recognition (OCR)). This is especially useful in applications that convert handwriting into LaTeX typesetting. Oftentimes, people must scan handwritten documents and digitize them, so high-quality handwriting recognition is vital. More specifically, college students, researchers, and professors may need to convert their handwritten work into plaintext in typesetting systems, such as LaTeX. Improving the quality of handwriting OCR can make this process more efficient.

Initially, we were planning to cover the entire scope of converting handwriting to formatted LaTeX typesetting, from OCR to combining the classified symbols in-order to create the LaTeX typesetting. However, due to our group's lack of prior knowledge in the field and the scope of the initial problem being too large, we decided to narrow the scope of the project to specifically the OCR portion in regards to mathematical symbols.

## Methods
  While we initially had chosen a dataset with 100,000 image samples from Kaggle with alphanumeric symbols, Greek letters, math functions and operators included, we  realized that it was a too large a dataset to go through. It took such a long time to simply unzip every symbol folder that we thought that even preprocessing would take a very long time, so we decided to utilize a smaller dataset instead. This new dataset had numeric digits and simple math operators, and was only a size of 10,000 images, which made initial dataset manipulation a lot easier [6]. 
  
  For the Logistic Regression model, after getting the initial dataset of images, we first cleaned the images. Although the handwriting images looked black and white (black ink, white background) to the naked eye, when we turned the images into arrays, they turned out to be colored images, as the image arrays were 3 dimensional, not 2 dimensional. Our solution for this was to greyscale the images so that they became black and white, and therefore 2 dimensional. We also resized the images to 40 x 40 pixels so that they would all become uniform in size, and also slighly make them smaller from their initial sizes. After this, we turned the images into corresponding 2D numpy arrays, flattened them all into 1D arrays and then stacked them to make a single large array for all of our images for our input array X. Finally, once we obtained a single input array, we then performed PCA on the array to reduce the dimensions to the first 150 components. This would serve to make building the models both easier and faster. After PCA, we performed Logistic Regression on the reduced images array.
  
  For the Convolutional Neural Network (CNN), we first defined our training and testing datasets using the "image_dataset_from_directory" function from tensorflow.keras, which allowed us to define the validation split we wanted to use in our data. Following popular practice, we chose 80/20 split for our data by setting validation_split to 0.2, which means 80% of our data is used for training and the other 20% for testing. In addition, we also used buffered pre-fetching in feeding our training and testing datasets to the model. This practice overlaps data pre-processing with model execution while training, which helps the model process the incoming dataset efficiently without the I/O being blocked. Finally, in our CNN model itself, some things we added for improved performance include an initial rescaling layer, which helps standardize the data for more accurate and quicker training, and multiple dropout layers, which ensured that our model did not overfit.

## Results and Discussion

### Logistic Regression Model

  Initially, when we ran the Logistic Regression Model on the array with the deafult settings from sklearn, we ran into issues at the fit function. No matter what we did to the image array, we kept on getting a non convergence error. We attempted to increase the number of max iterations rate from the initial 400 all the way to 11000, but we kept on failing to get the model to converge. This meant that the model was not fitting, and we could not obtain any meaningful data from our model.

![ClassificationReport](/docs/assets/classreport.png)

Then, we tried editing the C parameter of our model, as it was previously at its default value of 1. For reference, the C parameter (also known as a hyperparameter) represents how much the model should trust the training data given. At first, a lower value of C was set for the model (i.e., C = 0.1), which meant the model did not trust the training data too much and a high amount of regularization and overfitting. This, however, came to no avail, as the model still did not converge at all. This was when we tried increasing the C parameter to 10, which meant our model trusted our training data more and a lower level of regularization and overfitting. This caused our model to fully run and finally produce data.

Upon producing the classification report, which displayed precision, recall, and F1 score metrics for each label, it was evident that the Logistic Regression Model did not have as high of a performance as expected for the math handwriting dataset. Precision ranged from 0.3 to 0.6 for the different labels, with the '0' label having the lowest precision and the '+' symbol having the highest precision. It seemed to be a trend that simpler symbols such as the '+' sign and the decimal symbol had higher precision levels in general compared to more curvy or complex symbols. Similar trends were found for recall (ability for the model to detect positive samples) and F1 score (harmonic mean of precision and recall), with the decimal symbol having the highest recall (0.94) and F1 score (0.67).

![ClassificationReportHigherC](/docs/assets/class_model_higher_c_value.png)

Finally, we increased our C parameter to a even higher value of 1,000,000, which means that our model heavily relies and trusts the given training data. Overall, there were several labels that saw a slight increase in precision, recall and F1 score, especially in labels representing symbols with a more 'complex' shape. For instance, the 'z' label saw precision increase from 0.50 to 0.54, recall from 0.38 to 0.40, and F1 score from 0.43 to 0.46. This shows that for math-based handwriting, more accurate results come from trusting real-life training data more. However, the overall results did not change much.

Some of the reasons behind the performance of the Logistic Regression Model may be due to the smaller training set given (10,000 samples vs 100,000 samples) and the Logistic Regression Model not being as a good fit for math handwriting classification compared to other models.

### Convolutional Neural Network (CNN) Model
  After finishing working on our Logistic Regression Model, our group realized from doing research and receiving feedback that making a convolutional neural network (CNN) would be a more fitting model for classifying handwritten mathematical characters in our dataset. Initially, we started with 3 epochs (or iterations through the dataset) with a CNN model implementation similar to that found in Assignment 4. Once the model ran, we achieved a final validation accuracy of around 0.86. To increase the accuracy of our model, we then increased the number of epochs to 5, which enabled the model to achieve a final validation accuracy above 0.90. However, there were also clear signs of overfitting in our model based on the accuracy and loss graphs, so we decided to add dropout layers to our CNN model to counter the effects of overfitting by relying on more inputs as opposed to fewer inputs to our model. Once it was clear that our model was not overfitting anymore, we increased the amount of epochs to 20 for higher validation accuracy with the assurance that we would not be overfiting.
  
![CNNTrainingAccuracyGraph](/docs/assets/accuracy_graph_cnn.png)

![CNNTrainingLossGraph](/docs/assets/loss_graph_cnn.png)

  Once the state of the model was finalized, the final accuracy and validation accuracy achieved was 0.9949 and 0.9096, respectively. The final loss achieved was 0.0180 and 0.4927, respectively.
 
![CNNClassificationReport](/docs/assets/cnn_classification_report_ocr.png)

  We also analyzed the classification report that was generated from our CNN model. Precision values for the labels ranged from 0.76 to 1.00, recall from 0.82 to 0.98, and F1-score from 0.82 to 0.98. This means that the model is consistent and accurate in predicting where the handwritten math symbols should be classified. Like in the Logistic Regression model, simpler symbols such as the decimal and subtract sign has higher precision, recall, and F1-scores, whereas more symbols with more complex shapes such as the letter 'z' and 'x' are on the lower end for labels in terms of precision, recall and F1-score. Overall, with an average precision, recall, and F1-score of 0.91 among the 19 labels, which proves that the model is formidable and apt for the application.

![CNNConfusionMatrix](/docs/assets/CNNmatrix.png)

  In addition, we generated a multi-label, 19x19 confusion matrix from the CNN model. For context, any values across the diagonal of the confusion matrix signifies the number of handwritten characters properly classified as that particular label, and the other values off the diagonal means that the characters were mis-classified. The bluer the color of the box, the less number of characters were classified properly under that label; the more yellow the color of the box, more characters were classified properly under that label. Most of the labels have a warm, yellow-ish color, meaning a lot of characters were classified properly under it. In total, 1832 out of 2014 characters were classified properly in their respective labels, giving an evaluated accuracy of 90.963%.
  
## Conclusion

  Based on running our handwritten mathematical symbols dataset through two different models, the Logistical Regression and the Convolutional Neural Network (CNN) models, as well as comparing various model performance metrics such as accuracy, precision, recall, and F1-score, we can conclude that our CNN model performs far better in classification than the Logistical Regression model. This perfectly validates existing literature that claim that CNN's are ideal for image classification, especially with handwriting image OCR with mathematical symbols. With enough development, training, and tuning, this CNN model could be integrated as part of an application that effectively classifies and converts mathematical handwriting into LaTeX typesetting.
  
## Revised Timeline with Responsibilities
![Revised Gannt Chart](/docs/assets/ganttchart_midterm_update_math_ocr.png)

## References
[1] S. Pandey and A. Rohatgi, "Using OCR to automate document conversion to LATEX," 2021 International Conference on Computational Intelligence and Computing Applications (ICCICA), 2021, pp. 1-8, doi: 10.1109/ICCICA52458.2021.9697266. [https://ieeexplore.ieee.org/abstract/document/9697266](https://ieeexplore.ieee.org/abstract/document/9697266)


[2] Xuanxia Yao, T. G. (2021). A Deep Learning Technology based OCR Framework for Recognition Handwritten Expression and Text. CONVERTER, 01–10. https://doi.org/10.17762/converter.259

[3] Aly, W., Uchida, S., & Suzuki, M. (2009). Automatic classification of spatial relationships among mathematical symbols using geometric features. IEICE Transactions on Information and Systems, E92-D(11), 2235–2243. https://doi.org/10.1587/transinf.E92.D.2235 

[4] Clausner, C., Antonacopoulos, A., & Pletschacher, S. (2020). Efficient and effective OCR engine training. International Journal on Document Analysis and Recognition, 23(1), 73–88. https://doi.org/10.1007/s10032-019-00347-8

[5] S, Preetha, et al. “Machine Learning for Handwriting Recognition.” International Journal of Computer (IJC) , vol. 38, no. 1, 2020, pp. 93–101. [https://core.ac.uk/download/pdf/327266589.pdf](https://core.ac.uk/download/pdf/327266589.pdf)

[6] "Handwritten Math Symbols", Kaggle. (2021, May) [https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols?select=dataset](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols?select=dataset)
