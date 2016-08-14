Overview
--------

This study reviews exercise data from a variety of devices such as
Jawbone Up, Nike FuelBand, and Fitbit. The goal of this study is to
predict the manner in which the exercise participants did their
exercise. This is denoted by the "classe" variable in the training set.
Data for this study comes from
<http://groupware.les.inf.puc-rio.br/har>.

Training data is available at this link:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

Test data is available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

Loading Necessary Libraries
---------------------------

Before loading the data, I'll load the necessary R packages required for
my analysis.

    library (caret)
    library (knitr)
    library(rpart)
    library(randomForest)

I'll also set the seed to ensure reproducability.

    set.seed(55555)

Loading the Data
----------------

Before I can do any analysis, I'll load the training and test datasets.

    ## Setting the URLs for each dataset
    TrainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    TestURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    ## Reading the data
    trainData <- read.csv(url(TrainURL))
    testData <- read.csv(url(TestURL))

Now I'll partition the training data into a training and test set. The
original test set is set aside for the end, where I'll use the best
prediction model to predict the values for the Course Project Prediction
Quiz.

    ## Creating a training and test set from the training data
    ## 70% of the data will be used in the training set

    inTrain <- createDataPartition(trainData$classe, p=0.7, list=FALSE)
    trainSet <- trainData[inTrain,]
    testSet <- trainData[-inTrain,]

    dim(trainSet)

    ## [1] 13737   160

    dim(testSet)

    ## [1] 5885  160

Cleaning the Data
-----------------

The partitioned datasets each have 160 variables. Some of these
variables consist of mostly NA values. I'll remove the variables with
greater than 95% NA values to improve my analysis. I'll also remove the
Near Zero Variance variables and the first ID variable so that it
doesn't interfere with the prediction algorithms.

    ##Remove Near Zero Variance variables
    NZV <- nearZeroVar(trainSet)
    trainSet<-trainSet[,-NZV]
    testSet<-testSet[,-NZV]

    ## Remove mostly NA variables and the ID variable
    NAvalues <- sapply(trainSet, function(x) mean(is.na(x))>.95)
    trainSet <- trainSet[, NAvalues==FALSE]
    trainSet <- trainSet[c(-1)]
    testSet  <- testSet[, NAvalues==FALSE]
    testSet <- testSet[c(-1)]
    dim(trainSet)

    ## [1] 13737    58

    dim(testSet)

    ## [1] 5885   58

After removing the NA values, the Near Zero Variance variables, and the
ID variable, the data now consists of 58 variables.

Prediction Models
-----------------

The three different modeling alogirthms that I use here in my analysis
are as follows:

1.  Decision Trees
2.  Random Forest
3.  Generalized Boosted Model

Here I will create each of these three models. Note that I have already
set the seed value for my analysis.

### Decision Trees

Here is my code for building the Decision Trees Model. First I'll set
the cross validation to K = 3. Note that this K value will be used for
my other models as well.

    cvControl <- trainControl(method='cv', number = 3, verboseIter = FALSE)

    DTreesModel <- train(classe ~ ., data=trainSet,trControl=cvControl, method='rpart')

I'll check the out of sample error for the Decision Trees model to
determine prediction accuracy.

    predDTrees <- predict(DTreesModel, newdata=testSet)
    cMatDTrees <- confusionMatrix(predDTrees, testSet$classe)
    cMatDTrees

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1385  598  190  383   89
    ##          B    0    0    0    0    0
    ##          C  280  541  836  581  535
    ##          D    0    0    0    0    0
    ##          E    9    0    0    0  458
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4552          
    ##                  95% CI : (0.4424, 0.4681)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.2974          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8274   0.0000   0.8148   0.0000  0.42329
    ## Specificity            0.7008   1.0000   0.6014   1.0000  0.99813
    ## Pos Pred Value         0.5236      NaN   0.3015      NaN  0.98073
    ## Neg Pred Value         0.9108   0.8065   0.9389   0.8362  0.88483
    ## Prevalence             0.2845   0.1935   0.1743   0.1638  0.18386
    ## Detection Rate         0.2353   0.0000   0.1421   0.0000  0.07782
    ## Detection Prevalence   0.4494   0.0000   0.4712   0.0000  0.07935
    ## Balanced Accuracy      0.7641   0.5000   0.7081   0.5000  0.71071

We can see that the accuracy of the Decision Trees model against the
test data set is only 45.52%, which is relatively low. I'll now do the
same for the Random Forest Model.

### Random Forest

Here is my code for building the Random Forest Model.

    RFModel <- train(classe ~ ., data=trainSet,trControl=cvControl, method='rf')

I'll check the out of sample error for the Random Forest model to
determine prediction accuracy.

    predRF <- predict(RFModel, newdata=testSet)
    cMatRF <- confusionMatrix(predRF, testSet$classe)
    cMatRF

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    2    0    0    0
    ##          B    0 1136    2    0    0
    ##          C    0    1 1024    1    0
    ##          D    0    0    0  963    0
    ##          E    0    0    0    0 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.999           
    ##                  95% CI : (0.9978, 0.9996)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9987          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9974   0.9981   0.9990   1.0000
    ## Specificity            0.9995   0.9996   0.9996   1.0000   1.0000
    ## Pos Pred Value         0.9988   0.9982   0.9981   1.0000   1.0000
    ## Neg Pred Value         1.0000   0.9994   0.9996   0.9998   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1930   0.1740   0.1636   0.1839
    ## Detection Prevalence   0.2848   0.1934   0.1743   0.1636   0.1839
    ## Balanced Accuracy      0.9998   0.9985   0.9988   0.9995   1.0000

We can see that the accuracy of the Random Forest model against the test
data set is 99.9%. Since this is so high, I'll likely use the Random
Forest model against the final test set. To be thorough though, I'll do
the same analysis for a Generalized Boosted Model.

### Generalized Boosted Model

Here is my code for building the Generalized Boosted Model

    GBMModel <- train(classe ~ ., data=trainSet,trControl=cvControl, method='gbm')

I'll check the out of sample error for the Generalized Boosted model to
determine prediction accuracy.

    predGBM <- predict(GBMModel, newdata=testSet)
    cMatGBM <- confusionMatrix(predGBM, testSet$classe)
    cMatGBM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    4    0    0    0
    ##          B    1 1133    2    0    0
    ##          C    0    1 1022    2    0
    ##          D    0    1    2  962    2
    ##          E    0    0    0    0 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9975          
    ##                  95% CI : (0.9958, 0.9986)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9968          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9947   0.9961   0.9979   0.9982
    ## Specificity            0.9991   0.9994   0.9994   0.9990   1.0000
    ## Pos Pred Value         0.9976   0.9974   0.9971   0.9948   1.0000
    ## Neg Pred Value         0.9998   0.9987   0.9992   0.9996   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1925   0.1737   0.1635   0.1835
    ## Detection Prevalence   0.2850   0.1930   0.1742   0.1643   0.1835
    ## Balanced Accuracy      0.9992   0.9971   0.9977   0.9985   0.9991

We can see that the accuracy of the Generalized Boosted model against
the test data set is 99.75%. While this is still very high, it is not as
good as the 99.99% accuracy of the Random Forest model.

Running Prediction Model on Test Data
-------------------------------------

To recap, the accuracy of my predictions models was as follows:

1.  Decision Trees: 45.52%
2.  Random Forest: 99.9%
3.  Generalized Boosted Model: 99.75%

Based on this, the accuracy of the Random Forest model is highest and
therefore, I'll use that model to predict the exercise type (classe) of
the Test Data for the 20 quiz results. This prediction is shown below.

    TestPred <- predict(RFModel,newdata=testData)
    TestPred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
