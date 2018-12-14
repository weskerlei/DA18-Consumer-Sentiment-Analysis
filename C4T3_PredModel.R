##Import libraries
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(corrplot)
library(plotly)
library(kknn)

##Parallel Computing##
#--- for Win ---#
library(doParallel) 

# Check number of cores and workers available 
detectCores()
getDoParWorkers()
cl <- makeCluster(detectCores()-1, type='PSOCK')
registerDoParallel(cl)

##Import data sets
iphone_raw <- read.csv("iphone_smallmatrix_labeled_8d.csv")
galaxy_raw <- read.csv("galaxy_smallmatrix_labeled_9d.csv")


##Preprocessing 1 - Examine data set for iphone
summary(iphone_raw)
str(iphone_raw)
plot_ly(iphone_raw, x= ~iphone_raw$iphonesentiment, type='histogram')
  ## No missing value from iphone_raw from summary

###Preprocessing 2 - Feature selection###

##Examine Correlation matrix on iphone_raw##
iphone_cor <- cor(iphone_raw)
corrplot(iphone_cor)
    ##no features are highly correlated with dependent variable

## NZV - Examine feature variance using nearZeroVar with saveMetrics = TRUE
nzv_iMetrics <- nearZeroVar(iphone_raw, saveMetrics = TRUE)
nzv_iMetrics
nzv_iphone <- nearZeroVar(iphone_raw, saveMetrics = FALSE)
nzv_iphone
## NZV - create a new data set and remove near zero variance features
iphone_NZV <- iphone_raw[,-nzv_iphone]
iphone_NZV$iphonesentiment <- as.factor(iphone_NZV$iphonesentiment)
str(iphone_NZV)


## Recursive feature elimination##
#sample the data before using RFE
## set iphone sentiments as factors
set.seed(123)
iphone_raw$iphonesentiment <- as.factor(iphone_raw$iphonesentiment)
iphoneSample <- iphone_raw[sample(1:nrow(iphone_raw), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"))

## create new RFE dataset with RFE features
iphone_RFE <- iphone_raw[, predictors(rfeResults)]
  ##add dependent variable to iphone_RFE
iphone_RFE$iphonesentiment <- iphone_raw$iphonesentiment
str(iphone_RFE)

##Training 3 datasets on 4 models##
## 1 iphone_raw
inTrain1 <- createDataPartition(iphone_raw$iphonesentiment,
                               p=0.70,
                               list = FALSE)
training1 <- iphone_raw[inTrain1, ]
testing1 <- iphone_raw[-inTrain1, ]
nrow(training1)
nrow(testing1)

## 1.1 C5.0 on iphone_raw, 10 fold CV, tune length = 3
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(C50_Fit1 <- train(iphonesentiment ~.,
                             data = training1,
                             method = 'C5.0',
                             tuneLength = 3,
                             trControl = ctrl))
C50_Fit1

## 1.2 Random Forest, 10 fold CV, auto grid, tune length 5
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit1 <- train(iphonesentiment ~.,
                             data = training1,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit1

## 1.3 SVM, 10 fold CV, auto tune, tune length 3
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(svm_Fit1 <- train(iphonesentiment ~.,
                              data = training1,
                              method = 'svmPoly',
                              tuneLength = 3,
                              trControl = ctrl))
svm_Fit1

## 1.4 kknn, 10 fold CV, auto grid, tune length 5
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(kknn_Fit1 <- train(iphonesentiment ~.,
                             data = training1,
                             method = 'kknn',
                             tuneLength = 5,
                             trControl = ctrl))
kknn_Fit1

## 1 iphone_raw - predict test set
C50_Pred1 <- predict(C50_Fit1, newdata = testing1)
summary(C50_Pred1)
rf_Pred1 <- predict(rf_Fit1, newdata = testing1)
summary(rf_Pred1)
svm_Pred1 <- predict(svm_Fit1, newdata = testing1)
summary(svm_Pred1)
kknn_Pred1 <- predict(kknn_Fit1, newdata = testing1)
summary(kknn_Pred1)

postResample(C50_Pred1,testing1$iphonesentiment)
postResample(rf_Pred1,testing1$iphonesentiment)
postResample(svm_Pred1,testing1$iphonesentiment)
postResample(kknn_Pred1,testing1$iphonesentiment)


## 2 iphone_NZV
inTrain2 <- createDataPartition(iphone_NZV$iphonesentiment,
                                p=0.70,
                                list = FALSE)
training2 <- iphone_NZV[inTrain2, ]
testing2 <- iphone_NZV[-inTrain2, ]
nrow(training2)
nrow(testing2)

## 2.1 C5.0 on iphone_NZV, 10 fold CV, tune length = 3
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(C50_Fit2 <- train(iphonesentiment ~.,
                              data = training2,
                              method = 'C5.0',
                              tuneLength = 3,
                              trControl = ctrl))
C50_Fit2

## 2.2 Random Forest, 10 fold CV, auto grid, tune length 5
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit2 <- train(iphonesentiment ~.,
                             data = training2,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit2

## 2.3 SVM, 10 fold CV, auto tune, tune length 3
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(svm_Fit2 <- train(iphonesentiment ~.,
                              data = training2,
                              method = 'svmPoly',
                              tuneLength = 3,
                              trControl = ctrl))
svm_Fit2

## 2.4 kknn, 10 fold CV, auto grid, tune length 5
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(kknn_Fit2 <- train(iphonesentiment ~.,
                               data = training2,
                               method = 'kknn',
                               tuneLength = 5,
                               trControl = ctrl))
kknn_Fit2

## 2 iphone_NZV - predict test set with C5.0 and Random Forest only
C50_Pred2 <- predict(C50_Fit2, newdata = testing2)
postResample(C50_Pred2,testing2$iphonesentiment)

rf_Pred2 <- predict(rf_Fit2, newdata = testing2)
postResample(rf_Pred2,testing2$iphonesentiment)

## 3 iphone_RFE
inTrain3 <- createDataPartition(iphone_RFE$iphonesentiment,
                                p=0.70,
                                list = FALSE)
training3 <- iphone_RFE[inTrain3, ]
testing3 <- iphone_RFE[-inTrain3, ]
nrow(training3)
nrow(testing3)

## 3.1 C5.0 on iphone_RFE, 10 fold CV, tune length = 3
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(C50_Fit3 <- train(iphonesentiment ~.,
                              data = training3,
                              method = 'C5.0',
                              tuneLength = 3,
                              trControl = ctrl))
C50_Fit3

## 3.2 Random Forest, 10 fold CV, auto grid, tune length 5
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit3 <- train(iphonesentiment ~.,
                             data = training3,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit3

## 3.3 SVM, 10 fold CV, auto tune, tune length 3
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(svm_Fit3 <- train(iphonesentiment ~.,
                              data = training3,
                              method = 'svmPoly',
                              tuneLength = 3,
                              trControl = ctrl))
svm_Fit3

## 3.4 kknn, 10 fold CV, auto grid, tune length 5
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(kknn_Fit3 <- train(iphonesentiment ~.,
                               data = training3,
                               method = 'kknn',
                               tuneLength = 5,
                               trControl = ctrl))
kknn_Fit3

## 3 iphone_RFE - predict test set with C5.0 and Random Forest only
C50_Pred3 <- predict(C50_Fit3, newdata = testing3)
postResample(C50_Pred3,testing3$iphonesentiment)

rf_Pred3 <- predict(rf_Fit3, newdata = testing3)
postResample(rf_Pred3,testing3$iphonesentiment)

## Evaluate random forest and C5.0 on all 3 datasets
C50_cm1 <- confusionMatrix(C50_Pred1, testing1$iphonesentiment, positive = "1", mode = "everything") 
C50_cm1
rf_cm1 <- confusionMatrix(rf_Pred1, testing1$iphonesentiment, positive = "1", mode = "everything") 
rf_cm1

C50_cm2 <- confusionMatrix(C50_Pred2, testing2$iphonesentiment, positive = "1", mode = "everything") 
C50_cm2
rf_cm2 <- confusionMatrix(rf_Pred2, testing2$iphonesentiment, positive = "1", mode = "everything") 
rf_cm2

C50_cm3 <- confusionMatrix(C50_Pred3, testing3$iphonesentiment, positive = "1", mode = "everything") 
C50_cm3
rf_cm3 <- confusionMatrix(rf_Pred3, testing3$iphonesentiment, positive = "1", mode = "everything") 
rf_cm3
## from confusion matrix, observe that Class1, class 2, and class 4 performs poorly

## Feature Engineering by combining class labels in sentiments
## create a new dataset with 4 sentiment levels from iphone_raw
iphone_RC <- iphone_raw
iphone_RC$iphonesentiment <- recode(iphone_raw$iphonesentiment, "0" = "1", "1" = "1", "2" = "2", "3" = "3", "4" = "4", "5" = "4")
summary(iphone_RC)
str(iphone_RC)

## Train Random Forest on iphone_RC, 10 fold CV, auto grid, tune length 5
inTrain4 <- createDataPartition(iphone_RC$iphonesentiment,
                                p=0.70,
                                list = FALSE)
training4 <- iphone_RC[inTrain4, ]
testing4 <- iphone_RC[-inTrain4, ]
nrow(training4)
nrow(testing4)


ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit4 <- train(iphonesentiment ~.,
                             data = training4,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit4

## Evaluate random forest on iphone_RC performance
rf_Pred4 <- predict(rf_Fit4, newdata = testing4)
postResample(rf_Pred4,testing4$iphonesentiment)
rf_cm4 <- confusionMatrix(rf_Pred4, testing4$iphonesentiment, positive = "1", mode = "everything") 
rf_cm4

## create a new dataset with 4 sentiment levels from iphone_RFE
iphone_RC2 <- iphone_RFE
iphone_RC2$iphonesentiment <- recode(iphone_RFE$iphonesentiment, "0" = "1", "1" = "1", "2" = "2", "3" = "3", "4" = "4", "5" = "4")
summary(iphone_RC2)
str(iphone_RC2)

## Train Random Forest on iphone_RC2, 10 fold CV, auto grid, tune length 5
inTrain5 <- createDataPartition(iphone_RC2$iphonesentiment,
                                p=0.70,
                                list = FALSE)
training5 <- iphone_RC2[inTrain5, ]
testing5 <- iphone_RC2[-inTrain5, ]
nrow(training5)
nrow(testing5)


ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit5 <- train(iphonesentiment ~.,
                             data = training5,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit5

## Evaluate random forest on iphone_RC2 performance
rf_Pred5 <- predict(rf_Fit5, newdata = testing5)
postResample(rf_Pred5,testing5$iphonesentiment)
rf_cm5 <- confusionMatrix(rf_Pred5, testing5$iphonesentiment, positive = "1", mode = "everything") 
rf_cm5

## PCA - training1 and testing1 from iphone_raw (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(training1[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# use predict to apply pca parameters, create training, exclude dependant
train_pca <- predict(preprocessParams, training1[,-59])

# add the dependent to training
train_pca$iphonesentiment <- training1$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test_pca <- predict(preprocessParams, testing1[,-59])

# add the dependent to training
test_pca$iphonesentiment <- testing1$iphonesentiment

# inspect results
str(train_pca)
str(test_pca)

## Apply random forest to train_pca and test_pca
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit6 <- train(iphonesentiment ~.,
                             data = train_pca,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit6

## Evaluate random forest on iphone_pca performance
rf_Pred6 <- predict(rf_Fit6, newdata = test_pca)
postResample(rf_Pred6,test_pca$iphonesentiment)
rf_cm6 <- confusionMatrix(rf_Pred6, test_pca$iphonesentiment, positive = "1", mode = "everything") 
rf_cm6

## PCA + RC - using train and test set 4
preprocessParams4 <- preProcess(training4[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams4)

# use predict to apply pca parameters, create training, exclude dependant
train_pcaRC <- predict(preprocessParams4, training4[,-59])

# add the dependent to training
train_pcaRC$iphonesentiment <- training4$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test_pcaRC <- predict(preprocessParams, testing4[,-59])

# add the dependent to training
test_pcaRC$iphonesentiment <- testing4$iphonesentiment

# inspect results
str(train_pcaRC)
str(test_pcaRC)

## Apply random forest to train_pcaRC and test_pcaRC
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit7 <- train(iphonesentiment ~.,
                             data = train_pcaRC,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = ctrl))
rf_Fit7

## Evaluate random forest on iphone_pcaRC performance
rf_Pred7 <- predict(rf_Fit7, newdata = test_pcaRC)
postResample(rf_Pred7,test_pcaRC$iphonesentiment)
rf_cm7 <- confusionMatrix(rf_Pred7, test_pcaRC$iphonesentiment, positive = "1", mode = "everything") 
rf_cm7

## apply best performance model (RFE + RC) to large matrix
##Import data sets
iphoneLg_raw <- read.csv("iphoneLargeMatrix.csv")
## remove id feature from dataset
iphoneLg_raw$id <- NULL
summary(iphoneLg_raw)
str(iphoneLg_raw)

## Recursive feature elimination on iphone large matrix
#sample the data before using RFE
## set iphone sentiments as factors
set.seed(123)
iphoneLg_raw$iphonesentiment <- as.factor(iphoneLg_raw$iphonesentiment)

## create new RFE dataset with RFE features
iphoneLg_RFE <- iphoneLg_raw[, predictors(rfeResults)]
##add dependent variable to iphoneLg_RFE
iphoneLg_RFE$iphonesentiment <- iphoneLg_raw$iphonesentiment
str(iphoneLg_RFE)

## Apply random forest fit with RFE (rf_Fit5) to iphoneLg_RFE, predict 4 class
rf_LgPred <- predict(rf_Fit5, newdata = iphoneLg_RFE)
summary(rf_LgPred)
str(rf_LgPred)
## append prediction to final dataset iphoneLg_Pred
iphoneLg_Pred <- iphoneLg_RFE
iphoneLg_Pred$iphonesentiment <- rf_LgPred
summary(iphoneLg_Pred)
str(iphoneLg_Pred)

# create a data frame for plotting.
# you can add more sentiment levels if needed
# Replace sentiment values 
iphoneLg_pie <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive", "positive"), 
                           values = summary(rf_LgPred))
iphoneLg_pie
# create pie chart
plot_ly(iphoneLg_pie, labels = ~COM, values = ~values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

## visualize the iphone labeled data with 4 class from training set
# Replace sentiment values 
iphone_RC_pie <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive", "positive"), 
                      values = summary(iphone_RC$iphonesentiment))
iphone_RC_pie
# create pie chart
plot_ly(iphone_RC_pie, labels = ~COM, values = ~values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'iPhone Labeled Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))