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
library(ggplot)
library(parallel)
library(rpart)
library(dplyr)
library(caret)
```

```{r retrieveData}
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
na_strings <- c("", "#DIV/0!", "NA")    
training <- read.csv("data/pml-train.csv", row.names=1, 
                     na.strings=na_strings) %>% tbl_df()
testing <- read.csv("data/pml-testing.csv", row.names=1, 
                    na.strings=na_strings) %>% tbl_df()
```

## Data Cleanup

Exclude all columns that are mostly NA in value. Convert the `classe` 
outcome variable to a factor. We also remove the username and date/time columns 


```{r dataCleanup}
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

Candidate features after data cleansing are: `r names(training)`

## Partition Data

A new training and validation set is created using 70% of the original 
training set. A static random seed is set to ensure reproducibility.

```{r partitionData}
set.seed(8937)
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_clean <- training[inTrain,]
validation_clean <- training[-inTrain,]
```

## Identify Predictors
Use the `nearZeroVar` function of `caret` to identify features that have near 
zero variance and can be safely eliminated from the feature set. This gives us 
a more parsimonious model.

```{r reducePredictors}
nonusefullPredictors <- nearZeroVar(training_clean, saveMetrics = T)
eliminatedFeatures <- predictors[nonusefullPredictors$nzv]
usefullFeatures <- predictors[-nonusefullPredictors$nzv]
```

Eliminated `r length(eliminatedFeatures)` and preserved 
`r length(usefullFeatures)` for potential modelling.

# Train Model

The `doParallel` library is used to distribute model creation load across 
all CPU cores.

```{r, setupParallel}
library(doParallel)
registerDoParallel(detectCores())       #consume all available cores
```

Create a random forest model on our training data set using 4-fold cross 
validation. Note that this is processor intensive. To save rerun time, the 
model is serialized to disk and stored between runs, if possible.

```{r, createRFmodel}
set.seed(509475)
tc <- trainControl(method="cv", number=4)
if (file.exists("data/model_rf.Rdata")) {
                model_rf <- readRDS("data/model_rf.Rdata")
} else {
  model_rf <- train(classe ~ ., data=training_clean, model="rf", verbose=F, 
                    allowParallel=T, importance = T, trControl = tc)
  saveRDS(model_rf, file="data/model_rf.Rdata")         #save for faster resuse
}
```

# Evaluate Model
We take our random forest model and apply it against the validation hold out, 
generating a confusion matrix of predictions vs. actuals.
```{r, predictValidation}
validation_pred <- predict(model_rf, newdata=validation_clean)
confusionMatrix(validation_pred, validation_clean$classe)
```

Our accuracy with this model is 99.3%.

Look at the importance plot.
```{r, importancePlot}
# Check variable for importance
imp <- varImp(model_rf)$importance
varImpPlot(model_rf$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, 
           cex = 1, main = "Predictor Importance")
```

The top four variables for this model are `pitch_belt`, `yaw_belt`, `roll_belt`, and `magnet_dumbbell_z`.

# Final Predictions on Test Data Set
Create our final predictions on the testing data and 
output our answer files using the supplied function.

```{r, predictFinal}
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
