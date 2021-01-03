## Introduction,

_Finish later_

## Design, Participants, Measures:

_Use the previous paper._


## Data analysis:

**Re-sampling and Imputations:**

Same as the previous paper

**Removing noise from the beginning and the end of each activity**

Removed 600 readings, i.e. 20 seconds, from the beginning and the end of each activity. This eliminates the generated the noise when the participant is changing activity. (When participants switch between activities, we have a transient time which was removed earlier. However, as we have enough data, to assure that we don&#39;t introduce any noise from this process to our data we performed this second filtering process.




**Creating acceleration sequences:**

For image processing, a coloured image has three channels, and two dimensions. So for a 264\*264 pixel image,the shape is: 264, 264, 3

In our case, we used a window of 3 seconds. Considering the fact that the frequency is 30 Hz, the length of each sequence is 90. This is analogs to the number of pixels in one axis. Also we have 3 dimensions, z,y, and z similar to a coloured image that has 3 channels, so the each input sequence has a shape of 90 by 3. Now if we have n input (n sequesnces or n images), the input shape is (n,90,3).



**Demographic data preparation**

For the demographic data we extracted &#39;height&#39;,&#39;weight&#39;,&#39;age&#39;,&#39;gender&#39; and stored in separately to use them for some of the models. For each acceleration sequence, there is one row of demographic data which contains the participant&#39;s height (in centimetre), weight (in KG), age and gender. For female and male, the gender variable is 0 or 1 respectively.

**Labels** :

Each sequence is associated with an activity level. To prepare the data for our models, we converted the six different activity levels to categorical variables. This allows us to use a softmax activation function in the last layer of the model to estimate each activity level probability.

**Shuffling and splitting:**

For each wear location, we create 64726 sequences. These sequences are divided into training and test sets to train the model and validate it.

10 percent of the data was reserved for testing. The data is shuffled prior to modelling. ( _can explain later in the discussion that although we used 90% of the data to train, there&#39;s no over-fitting and the performance of the model on the test data is almost the same as the training data. Thus 9 to 1 ration for training and testing data was a good choice. Because we used more data to train the model)_


## Modelling

_Briefly explain convectional layers and recurrent layers._

_Then mention the models we use in one paragraph:_


Several model were designed to classify activity levels. Some of these models used only the acceleration data and others used both acceleration and demographic data. These models utilized a combination of convectional layers, LSTM layers and dense layers.

_Then mention each model specification. Can be in one single table. Ask which of the models should be in the final version of the paper._



**CNN model:**
![CNN Model Structure](https://github.com/khataei/PE-classification-DeepLearning/blob/master/CNN-model.png)

_Double check the performance of the models and hyperparameter tuning notebook. Select the best one and mentioned here._

**LSTM model:**

_Same as above_

**CNN and LSTM combination:**

_same as above_

**Demographic and CNN model:**

_Explain that we have two input and mention that first the acceleration sequence data is feed to a convectional layer and then the result is added to demographic data and all of them will be fed to a dense later._


## Results:

| Model    | Location | Accuracy | ROC   |
|----------|----------|----------|-------|
| CNN      | Pocket   | 96.31    | 99.73 |
| LSTM     | Pocket   | 95.20    | 99.56 |
| CNN-LSTM | Pocket   | 96.23    | 99.68 |
| CNN      | Hand     | 91.77    | 98.99 |
| LSTM     | Hand     | 90.16    | 98.33 |
| CNN-LSTM | Hand     | 91.26    | 98.73 |
| CNN      | Backpack | 93.19    | 99.17 |
| LSTM     | Backpack | 92.19    | 98.65 |
| CNN-LSTM | Backpack | 93.31    | 98.95 |

_After writing the modelling section finish this section. Use ROC and accuracy. Use a table to show the results._

Base model : 93% accuracy

CNN: 96% +

LSTM: 94 or 95 %

CNN + LSTM: 96%

Demographic + CNN: 97% +

_Question: Should we include hand and backpack location as well? Their analyses are done. The model structures are the same so the modelling section should remain the same. Only the result part would be bigger._



## Discussion:

_Finish after modelling and results._
