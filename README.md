# Hurricane_Prediction_Model

This project is consists of a machine learning pipline that detects hurricanes in infared satellite image and predicts the hurricane's path with an RNN. Specific information about the project is contained in the Jupyter Notebook. 

The contents consist of the following:
* Hurricane_Prediction.ipynb -- Jupyter Notebook that contains the code to run and detailed explainations of my methods.
* hurricane_data.csv -- sample data that could be loaded into a dataframe.
* sample_yolo_footage.mp4 -- sample footage that's processed by the YOLOv26 portion of the workflow.

To run said Jupyter Notebook, simply download it and upload the file into google collab.  

# Hurricane Dataset (COCO Format)
For those who are interested in the original dataset, you can download it in the COCO format from Kaggle: https://www.kaggle.com/datasets/bot5563/labelled-infrared-hurricane-2019-2024

# Project Explanation

### Purpose
Creating a ML project that uses yolo, bytetrack, and rnns to create a hurricane detection & prediction algorithm that could detect hurricanes based on infrared footage and accurately predict their future pathing

### How the General Workflow Works
YOLO (you only look once) model will initially detect when a hurricane forms
ByteTrack algorithm will output the number of hurricanes it detects, their size, and their relative velocities
A many to many RNN will take that data and use it predict the pathing of the hurricane and potentially when the hurricane disappears

### Creating the Dataset
All footage of hurricanes (that are infrared, ~15 minutes, long year-round) will be recorded from weather.gov’s redirect links. To get the data, I will be using the yt-dlp command to download the redirect links in a mp4 format. In addition, NodeJS 22 will be the javascript runtime environment to resolve the issues   

After that, I would upload the footage to roboflow and manually splice the footage and draw the bounding boxes. (I'm fully aware that there is an AI assistent that could help me, but unfortunately, it didn't perform well in my case).

### Videos Used In the Training Set
* 2024 — https://youtu.be/w2MM7BIpqYY
* 2023 — https://youtu.be/ANb9AQYFA94
* 2022 — https://youtu.be/hTmjRqykmdM
* 2021 — https://youtu.be/2yb0Ra5AZHs
* 2020 — https://youtu.be/Lthy2r_91_Q
* 2019 — https://youtu.be/XHHQEByoeCo

### Dataset and Image Detection Module

In total, my dataset contains 553 images with a 70/20/10 split on the training, validation, and test set. Thanks to Roboflow's automatic training features, this module has very solid performance:
* mAP@50 -- 93.1%
* Precision -- 85.2%
* Recall -- 88.2%
* F1 -- 86.7%

After that, I created a simple workflow with Yolo26 for image detection and ByteTrack Tracker for object ids & tracking images across frames, an essential feature for the prediction part of this project.

### Processor For Video Footage

Unfortunately, downloading weights from Roboflow would cost me a mind-boggling $99/month. Thus, I did the next best thing of deploying the model as an API. How the following code will work is that it will take a video file, split it into images, send the images to the roboflow workflow, and return data such as bounding boxes and object ids.

### Sliding Window for RNN Data Preparation

To prepare the data for a Recurrent Neural Network (RNN), we need to transform the time-series data into sequences. An RNN processes data sequentially, where each prediction depends on the previous inputs. The sliding window technique is commonly used for this:

1.  **Input Sequence (X)**: We define a `sequence_length` (e.g., 5 frames). For each hurricane, we take a window of `sequence_length` consecutive frames as input.
2.  **Target Sequence (y)**: Corresponding to each input sequence, we define a `prediction_horizon` (e.g., 1 frame). The target for the input sequence will be the `prediction_horizon` number of frames *immediately following* the input sequence.

This function will iterate through each `hurricane_id` in your DataFrame, applying this sliding window to create a dataset suitable for training your RNN. We'll be predicting the next `prediction_horizon` elements based on the `sequence_length` preceding elements.

In addition to the AI generated code, I made two adjustment. One, if there are less than 5 frames for an identified hurricane, the data will be discarded (since it's likely noise). Two, if there are less data in the prediction than the prediction horizon, I the prediction tensor with zeros (perfectly acceptable in my use case).

### RNN
This isn't an ordinary RNN. For this portion of the project, I'd like the RNN to predict both the future path of the hurricane and if the hurricane is dead or alive. (Initially, I planned to simply have the RNN predict [0,0,0,0] to signify a dead hurricane. However, I was advised against this since a change from say 40px to 0px width would probably cause an exploding gradient that messes up my model).

To make things efficient, this RNN will be a multi-task model that outputs a tuple (binary dead/alive, tensor on bounding boxes). There will be two heads: a binary classification one with sigmoid and regression. The former is responsible for predicting whether the hurricane would be alive, and the latter will predict the bounding box dimensions.

### Masked Multipliers Class
The class serves two purposes: a custom loss function for our rnn and holding two parameters that act as multipliers for the two losses. As standard, MSE loss will be used for the regression portion and BCE loss will be used for the binary classification portion. The tricky part is how to combine these losses to make sense. The algorithm used in this class is described in this paper about multi-task models: https://arxiv.org/pdf/1705.07115.

As stated in the paper, since these results come from different distributions (bernolli and gaussian), it's possible that the results output different ranges. If these ranges are unbalanced for, one distribution is going to contribute a lot more to the loss function and affect backpropagation. To compensate, one could use an approach to weigh and aggregate the loss values: e.g., a * bce-loss + b * mse-loss.

Finding such values a & b manually would be tedious. Also, using a model to find them wouldn't be ideal because said model could easily cheat the system by setting a & b to 0 or a negative value.

Thus, they proposed the following loss function: 1/(2sigma$^2_1$)L$_1$(W) + 1/(sigma$^2_2$)L$_2$(W) + log(sigma$_1$) + log(sigma$_2$).
* L$_1$(W) -- euclidean loss
* L$_2$(W) -- cross entropy loss
* a = log(sigma$^2_1$)
* b = log(sigma$^2_2$)
* sigma$^2$ = exp(a) or exp(b)
* log(sigma) = 0.5a or 0.5b

Thus, for the code, it converts a & b into
* 1/exp(-a) + 0.5*a
* 1/exp(-b) + 0.5*b

Although one could try to optimize for sigma directly instead of something like s = log(sigma$^2_1$), the paper stated that in practice, it is best to optimize for s = log(sigma$^2_1$) because it is more stable.

In this equation, it would be impossible for the model to cheat either sigma values (by setting them to zero or any small number) bc/ either the 1/(sigma$^2$) or log(sigma) parts of the equation would increase the loss dramatically.

Generally speaking, one can construct these custom loss functions through the following steps:
* use the appropriate loss function (MSE for regression, BCE for binary classification, and CE for multi classification)
* allocating a parameter to the loss, and applying
* create the loss equation using the following equation

$TotalLoss = sum_{i=1 to N} (C_i * exp(-s_i) * L_i(W) + 0.5 * s_i)$
* $L_i(W)$ -- output of the loss function
* s_i -- loss multiplier
* C_i -- constant factor (0.5 for classification & 1.0 for classification)

After computing the loss, we could call .backwards() on it like any other loss function output because PyTorch's autograd system would correct the gradients for us.

### Model Training Procedure
Our MaskedMultiTaskLoss is our loss function. As for the optimizer, the standard Adam should be good enough. An importants step is to pass in parameters from both our loss and rnn model to ensure all parameters are properly optimized. The rest of the code is just a standard training loop.

### Model Evaluation Procedure
Since the results should come from data distributions, I've decided to evaluate the results seperately using the standard approaches for both: r2 score for the regression portion and accuracy_score for the binary classification portion.

### Results and Conclusion
The above results you see are the workflow on a small clip of the 2019 Hurricane footage. I adjusted the model so that the RNN takes 5 sequences and attempts to predict the nxt 3 sequences. After 70 epochs, the R2 score is .71 and classification accuracy is 100%. Though admittedly, when used on the same dataset, if one tries to make the RNN input 10 sequences and attempts to predict the next 8 sequences, the R2 score will dramatically decrease to .48. That result is expected though because trying to predict 8 sequences while having a dataset of 470 is ambitious to say the least.

The reason that I didn't use the whole is due to the massive dataset size and cost of credits.

Just this 17 second footage created 460 tuples of data. If I fed all 6 of the 15 minute videos, I'd end up with a pretty big dataset. Consequently, such a huge upload may rapidly use up all my credits, which is an expense I'd rather avoid. In addition, just this 17 seconds of footage took ~4 minutes to finish processing. Doing the math, this means a 15 minute video would take more than 3 hours to finish processing.

I'm aware an alternative option is to download the yolo wieghts and use ultralytics; however, that would cost me $99. Plus, it took me 14 minutes to train the Roboflow workflow. So, this will be the more time efficient option.

But at least this decent performance show that this concept of using yolo+bytetrack to detect hurricanes in satellite image, and then using rnns to predict hurricane paths should work well in theory.

### Limitations
A major limitation about ML is that it’s very difficult for models to adapt to changing conditions. Thus, as global temperatures continue to rise, it’s possible that the RNN cannot keep up with these changes: leading to inaccurate predictions.

The model was trained on data from weather.gov, and several aspects of the data limit this system’s practicality: infrared only footage, specific angle of the footage, and limited footage. The videos are infrared, meaning that none-infrared data could lead to inaccurate predictions & detections. All of this footage is of one angle over the Atlantic Ocean. Consequently, predictions of hurricanes will only be accurate over this angle over the Atlantic Ocean specifically. I highly doubt that this model could predict storms in the Pacific or Indian Ocean well. Finally, this site only has data between 1978 and 2024. While that should be plenty for this project, a much larger dataset would improve accuracy significantly.

There is also the limitation of time where processing a 15 minute video for its bounding box outputs could take hours. However, if one has a lot of free time to spare, then this could be overcomed.

###Suggestions/Enhancements for Future Iterations of this Project
A possible idea to improve this project would be to input additional information in the RNN input to better predict hurricane paths (e.g., air pressure and sea surface temperature of the hurricane eye).

Perhaps one could also ensure that this RNN works globally by inputting the exact longitude and latitude as well and retraing the RNN, so that the RNN isn't stuck to predicting this one region over the Atlantic Ocean specifically. (Retraining the Yolo shouldn't be necessary bc/ all hurricanes kinda look the same).

If one is ambitious, perhaps instead of just predicting hurricane paths, one could try to predict when and where hurricanes would spawn in the world.

Finally, perhaps experiement RNN architectures. For instance, I've heard of something called a GRU that I shall further look into.
