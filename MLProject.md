---
title: "ML Project"
author: "Zhenyu Yang"
date: "03-27-2016"
output: 
  html_document: 
    keep_md: yes
---

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

```{r libraryLoad, message=FALSE, warning=FALSE}
library(ggplot2)
library(parallel)
library(doParallel)
library(rpart)
library(dplyr)
library(caret)
```

```{r retrieveData}
train_int <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_int  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("data")) 
  { dir.create("data") 
} 
if (!file.exists("data/pml-train.csv")) 
  {download.file(train_url, "data/pml-train.csv")
}
if (!file.exists("data/pml-testing.csv")) 
  {
  download.file(test_int, "data/pml-testing.csv")
}

# load data
nana <- c("", "#DIV/0!", "NA")    
training <- read.csv("data/pml-train.csv", row.names=1, na.strings=nana) %>% tbl_df()
testing <- read.csv("data/pml-testing.csv", row.names=1, na.strings=nana) %>% tbl_df()
```

## Data Cleanup

Exclude all columns that are mostly NA in value. 
Convert the `classe` outcome variable to a factor. 
Eliminate the username and date/time columns 


```{r dataCleanup}
training$classe <- as.factor(training$classe)

# variables for the predictor columns
numeric_cols <- sapply(training, is.numeric)
predictors <- names(training)[numeric_cols]

# get columns that are not all NAs
training <- training[,colSums(is.na(training))<nrow(training)]

# look for columns that are mostly NAs
na_test <- sapply(training, function(x) {sum(is.na(x))})

#table(na_test)

elimi_columns <- names(na_test[na_test>=19216])

#remove corelated variables
elimi_columns <- c(elimi_columns, "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window")

training = training[, !names(training) %in% elimi_columns]
```

Candidate features after data cleansing are: `r names(training)`

## Partition Data

A new training and validation set is created using 70% / 30% of the original 
training set. 

A random seed is set to ensure reproducibility.

```{r partitionData}
set.seed(8607)
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_clean <- training[inTrain,]
validation_clean <- training[-inTrain,]
```

## Identify Predictors
Use the `nearZeroVar` function of `caret` to identify features that have near zero variance. Those variables can be safely eliminated from the feature set. 

```{r reducePredictors}
nonusefullPredictors <- nearZeroVar(training_clean, saveMetrics = T)
eliminatedFeatures <- predictors[nonusefullPredictors$nzv]
usefullFeatures <- predictors[-nonusefullPredictors$nzv]
```

Eliminated `r length(eliminatedFeatures)` and preserved 
`r length(usefullFeatures)` for potential modelling.

## Train Model

The `doParallel` library is used to distribute workload across all CPU cores.

```{r, setupParallel}
registerDoParallel(detectCores())     
```

Build Random Forest model on our training data set using 4-fold cross validation. 

```{r, createRFmodel}
set.seed(509475)
tc <- trainControl(method="cv", number=4)
if (file.exists("data/model_rf.Rdata")) {
                model_rf <- readRDS("data/model_rf.Rdata")
} else {
  model_rf <- train(classe ~ ., data=training_clean, model="rf", verbose=F, 
                    allowParallel=T, importance = T, trControl = tc)
  saveRDS(model_rf, file="data/model_rf.Rdata")         
}
```

## Evaluate Model
Take the random forest model and apply it on the validation set.
Generate a confusion matrix of predictions vs. actuals.

```{r, predictValidation}
vali_pred <- predict(model_rf, newdata=validation_clean)
confusionMatrix(vali_pred, validation_clean$classe)
```

The accuracy here in this model is 99.73%.

Look at the importance plot.

```{r, importancePlot}
# Check variable for importance
imp <- varImp(model_rf)$importance
varImpPlot(model_rf$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, 
           cex = 1, main = "Predictor Importance")
```

The top four important variables for RF model are `yaw_belt`, `pitch_belt`,  `roll_belt`, and `magnet_dumbbell_z`.

## Final Predictions
Apply final predictions on the testing data and output the answer files using the supplied function.

```{r, predictFinal}
predictions <- predict(model_rf, testing)
#confusionMatrix(predictions, testing$classe)

write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename  <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}

write_files(predictions)
```
