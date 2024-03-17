---
title: "Prediction Assignment Writeup"
author: "Heidi Tang"
date: "3/17/24"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

### 1 Executive Summary

People often quantify how much of a certain task that they work on, but tend to ignore how well they finish it. This report tries to predict the manner in which they did the exercise. The data are from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The target variable `classe` is in the training set, and other variables are used for predictions.

Through cross validation (CV) at a training-test data split ratio of 7:3, the final "best" model was found in three steps. First, remove useless variables in terms of accelerometer reading and those with many NA values. As a result, only 53 variables remained. Second, apply a decision tree (DT) model to predict the targeted manner. With this model, the expected out of sample error reached 50%, which was very high. Last, use random forests (RF) with 3-fold CV to conduct the same prediction. Compared to the DT, the expected out of sample error of this model was much lower, with only 1%, which implied that the RF model performed much better than the DT model. Therefore, the RF model was chosen to predict the manner. 

Besides, the selected RF model was applied to predict 20 different test cases.

#### Load libraries
```{r results="hide", warning=FALSE, error=FALSE, message=FALSE}
library(caret)
library(rpart)
library(rattle)
library(scales)
library(randomForest)
```

### 2 Training vs. Test Split

#### Set seed
```{r results="hide", warning=FALSE, error=FALSE, message=FALSE}
set.seed(1357)
```

#### Load training and test data
```{r}
train_data_url = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
test_data_url = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

if (file.exists('pml-training.csv') == FALSE) {
  download.file(train_data_url, 'pml-training.csv')
}
if (file.exists('pml-testing.csv') == FALSE) {
  download.file(test_data_url, 'pml-testing.csv')
}

train <- read.csv('pml-training.csv', na.strings=c("","NA"))
test <- read.csv('pml-testing.csv', na.strings=c("","NA"))
```

#### CV with tra.: test data = 0.7: 0.3
The data are split into (1) training data and (2) test data (for cross validation).
```{r}
inTrain <- createDataPartition(train$classe, p=.7, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]

summary(training$classe)
```
    ## Length     Class      Mode 
    ##  13737 character character 
    
#### Clean data
```{r}
# Remove useless time-related & recording variables, and the row index variable X.
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
test <- test[, -c(1:7)]
```

```{r}
# Removed variables with many missing values. NAs and blank fields are marked as NA.
mostlyNAs <- which(colSums(is.na(training)) > nrow(training)/2)
training <- training[, -mostlyNAs]
testing <- testing[, -mostlyNAs]
test <- test[, -mostlyNAs]
```

### 3 Modeling: Decision Trees vs. Random Forests

#### 3.1 Decision Trees
```{r}
# Train decision tree model
rpModelFit <- train(classe ~ ., method="rpart", data=training)
rpModelFit$finalModel
```
    ## n= 13737 

    ## node), split, n, loss, yval, (yprob)
         ##  * denotes terminal node

     ## 1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
       ## 2) roll_belt< 130.5 12581 8685 A (0.31 0.21 0.19 0.18 0.11)  
         ## 4) pitch_forearm< -33.55 1112    8 A (0.99 0.0072 0 0 0) *
         ## 5) pitch_forearm>=-33.55 11469 8677 A (0.24 0.23 0.21 0.2 0.12)  
          ## 10) magnet_dumbbell_y< 432.5 9592 6872 A (0.28 0.18 0.24 0.19 0.11)  
            ## 20) roll_forearm< 122.5 5951 3523 A (0.41 0.18 0.19 0.17 0.059) *
            ## 21) roll_forearm>=122.5 3641 2437 C (0.08 0.18 0.33 0.23 0.18) *
          ## 11) magnet_dumbbell_y>=432.5 1877  948 B (0.038 0.49 0.045 0.23 0.19) *
       ## 3) roll_belt>=130.5 1156   10 E (0.0087 0 0 0 0.99) *

```{r}
# Plot the model
fancyRpartPlot(rpModelFit$finalModel, sub='')
```
![](https://github.com/Heidi24/Prediction-Assignment-Writeup/blob/main/fig1.png)


```{r}
# Predict `classe` for test data
rpPreds <- predict(rpModelFit, testing)
rpConMatrix <- confusionMatrix(rpPreds, as.factor(testing$classe))
rpConMatrix
```

    ## Confusion Matrix and Statistics

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1526  466  473  423  165
    ##          B   35  391   31  171  138
    ##          C  109  282  522  370  294
    ##          D    0    0    0    0    0
    ##          E    4    0    0    0  485

    ## Overall Statistics
                                         
    ##                Accuracy : 0.4969         
    ##                  95% CI : (0.484, 0.5097)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
                                         
    ##                   Kappa : 0.3425         
                                         
    ##  Mcnemar's Test P-Value : NA             

    ## Statistics by Class:

    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9116  0.34328   0.5088   0.0000  0.44824
    ## Specificity            0.6374  0.92099   0.7829   1.0000  0.99917
    ## Pos Pred Value         0.4998  0.51044   0.3310      NaN  0.99182
    ## Neg Pred Value         0.9477  0.85388   0.8830   0.8362  0.88936
    ## Prevalence             0.2845  0.19354   0.1743   0.1638  0.18386
    ## Detection Rate         0.2593  0.06644   0.0887   0.0000  0.08241
    ## Detection Prevalence   0.5188  0.13016   0.2680   0.0000  0.08309
    ## Balanced Accuracy      0.7745  0.63213   0.6458   0.5000  0.72371

```{r}
# Accuracy Evaluation
rpAccuracy = rpConMatrix$overall[[1]]
percent(rpAccuracy)
```

    ## [1] "50%"
Low accuracy in the DT model.

##### Expected Out of Sample Error
```{r}
percent(1.00-rpAccuracy)
```
    ## [1] "50%"

#### 3.2 Random Forests
```{r}
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
rfModelFit <- train(classe ~., method="rf", data=training, trControl=fitControl)
rfModelFit$finalModel
```

    ## Call:
     ## randomForest(x = x, y = y, mtry = param$mtry) 
                   ## Type of random forest: classification
                         ## Number of trees: 500
    ## No. of variables tried at each split: 27

            ## OOB estimate of  error rate: 0.72%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3900    4    0    0    2 0.001536098
    ## B   20 2631    7    0    0 0.010158014
    ## C    0   11 2376    9    0 0.008347245
    ## D    0    1   29 2217    5 0.015541741
    ## E    0    1    3    7 2514 0.004356436

```{r}
# Predict `classe` for test data
rfPreds <- predict(rfModelFit, newdata=testing)
rfConMatrix <- confusionMatrix(rfPreds, as.factor(testing$classe))
rfConMatrix
```

    Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1667   12    0    0    0
         B    4 1125    8    0    0
         C    3    1 1016   12    0
         D    0    1    2  952    2
         E    0    0    0    0 1080

Overall Statistics
                                          
               Accuracy : 0.9924          
                 95% CI : (0.9898, 0.9944)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9903          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9958   0.9877   0.9903   0.9876   0.9982
Specificity            0.9972   0.9975   0.9967   0.9990   1.0000
Pos Pred Value         0.9929   0.9894   0.9845   0.9948   1.0000
Neg Pred Value         0.9983   0.9971   0.9979   0.9976   0.9996
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2833   0.1912   0.1726   0.1618   0.1835
Detection Prevalence   0.2853   0.1932   0.1754   0.1626   0.1835
Balanced Accuracy      0.9965   0.9926   0.9935   0.9933   0.9991

```{r}
# Accuracy Evaluation
rfAccuracy = rfConMatrix$overall[[1]]
percent(rfAccuracy)
```

     ## [1] "99%"
     
Higher accuracy in the RF model than in the DT model.

##### Expected Out of Sample Error
```{r}
percent(1.00-rfAccuracy)
```

     ## [1] "1%"

### 4 Conclusion
The RF model was chosen for a better performance than the DT model.

### 5 Predictions in 20 Test Cases
```{r}
submissionPreds <- predict(rfModelFit, newdata=test)
submissionPreds
```
     ## [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
