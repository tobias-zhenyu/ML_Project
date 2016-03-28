# ML Project
Zhenyu Yang  
03-27-2016  

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:

Class | Description
------|------------
Class A | Exactly according to the specification
Class B | Throwing the elbows to the front
Class C | Lifting the dumbbell only halfway
Class D | Lowering the dumbbell only halfway
Class E | Throwing the hips to the front

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Data

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.  


```r
library(dplyr)
library(caret)
```


```r
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("data")) { dir.create("data") } 
if (!file.exists("data/pml-train.csv")) {
  download.file(train_url, "data/pml-train.csv")
}
if (!file.exists("data/pml-testing.csv")) {
  download.file(test_url, "data/pml-testing.csv")
}

# load data
na_strings <- c("", "#DIV/0!", "NA")    #consider these strings as NAs
training <- read.csv("data/pml-train.csv", row.names=1, 
                     na.strings=na_strings) %>% tbl_df()
testing <- read.csv("data/pml-testing.csv", row.names=1, 
                    na.strings=na_strings) %>% tbl_df()
```

## Data Cleanup

Eliminate all columns that are mostly NA in value. Convert the `classe` 
outcome variable to a factor. We also remove the username and date/time columns 
as we are not considering time to be a factor in this model.


```r
training$classe <- as.factor(training$classe)

# create variables for our outcome and numeric predictor columns
numeric_cols <- sapply(training, is.numeric)
predictors <- names(training)[numeric_cols]

# easy elimination, get columns that are not all NAs
training <- training[,colSums(is.na(training))<nrow(training)]

# look for columns that are more mostly NAs
na_test <- sapply(training, function(x) {sum(is.na(x))})
#table(na_test)

bad_columns <- names(na_test[na_test>=19216])

#also remove some confounding variables
bad_columns <- c(bad_columns, "user_name", "raw_timestamp_part_1", 
                 "raw_timestamp_part_2", "cvtd_timestamp", "num_window")

training = training[, !names(training) %in% bad_columns]
```

Candidate features after data cleansing are: new_window, roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm, total_accel_arm, gyros_arm_x, gyros_arm_y, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, magnet_arm_x, magnet_arm_y, magnet_arm_z, roll_dumbbell, pitch_dumbbell, yaw_dumbbell, total_accel_dumbbell, gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z, accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z, magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z, roll_forearm, pitch_forearm, yaw_forearm, total_accel_forearm, gyros_forearm_x, gyros_forearm_y, gyros_forearm_z, accel_forearm_x, accel_forearm_y, accel_forearm_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z, classe

## Partition Data

A new training and validation set is created using 70% of the original 
training set. A static random seed is set to ensure reproducibility.


```r
set.seed(1337)
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_clean <- training[inTrain,]
validation_clean <- training[-inTrain,]
```

## Identify Predictors
Use the `nearZeroVar` function of `caret` to identify features that have near 
zero variance and can be safely eliminated from the feature set. This gives us 
a more parsimonious model.


```r
nonusefullPredictors <- nearZeroVar(training_clean, saveMetrics = T)
eliminatedFeatures <- predictors[nonusefullPredictors$nzv]
usefullFeatures <- predictors[-nonusefullPredictors$nzv]
```

Eliminated 3 and preserved 
148 for potential modelling.

# Train Model

The `doParallel` library is used to distribute model creation load across 
all CPU cores.


```r
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```r
registerDoParallel(detectCores())       #consume all available cores
```

Create a random forest model on our training data set using 4-fold cross 
validation. Note that this is processor intensive. To save rerun time, the 
model is serialized to disk and stored between runs, if possible.


```r
set.seed(57475)
tc <- trainControl(method="cv", number=4)
if (file.exists("data/model_rf.Rdata")) {
                model_rf <- readRDS("data/model_rf.Rdata")
} else {
  model_rf <- train(classe ~ ., data=training_clean, model="rf", verbose=F, 
                    allowParallel=T, importance = T, trControl = tc)
  saveRDS(model_rf, file="data/model_rf.Rdata")         #save for faster resuse
}
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

# Evaluate Model
We take our random forest model and apply it against the validation hold out, 
generating a confusion matrix of predictions vs. actuals.

```r
validation_pred <- predict(model_rf, newdata=validation_clean)
confusionMatrix(validation_pred, validation_clean$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    9    0    0    0
##          B    0 1127   13    0    0
##          C    0    3 1013   14    0
##          D    0    0    0  950    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9873   0.9855   0.9982
## Specificity            0.9979   0.9973   0.9965   0.9996   1.0000
## Pos Pred Value         0.9947   0.9886   0.9835   0.9979   1.0000
## Neg Pred Value         1.0000   0.9975   0.9973   0.9972   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1915   0.1721   0.1614   0.1835
## Detection Prevalence   0.2860   0.1937   0.1750   0.1618   0.1835
## Balanced Accuracy      0.9989   0.9934   0.9919   0.9925   0.9991
```

Our accuracy with this model is 99.3%.

Look at the importance plot.

```r
# Check variable for importance
imp <- varImp(model_rf)$importance
varImpPlot(model_rf$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, 
           cex = 1, main = "Predictor Importance")
```

![](ML_files/figure-html/importancePlot-1.png)

The top four variables for this model are `pitch_belt`, `yaw_belt`, `roll_belt`, and `magnet_dumbbell_z`.

# Final Predictions on Test Data Set
Create our final predictions on the testing data and 
output our answer files using the supplied function.


```r
predictions <- predict(model_rf, testing)
#confusionMatrix(predictions, testing$classe)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename  <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE, 
                row.names=FALSE, col.names=FALSE)
  }
}

pml_write_files(predictions)
```
