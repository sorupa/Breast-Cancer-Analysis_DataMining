# breast Cancer disease classification with Neural Network Model & SVM:

# load the library
library(mlbench)
library(caret)
library(dplyr)
library(corrplot)
library(nnet) 
library(NeuralNetTools)
library(e1071)


# url for Breast Cancer data from Wisconsin

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# rename column names made shorter for plotting of nnet model

# CPTH  clump.thick       CLSZ  cell.size 
# CLSP  cell.shape        MADH  mar.adhesion
# EPTH  epithelial        BNUC  bare.nuclei
# BLCR  bland.chromatin   NNUC  normal.nuclei 
# MTSE  mitoses           class tumour type

breast_cancer_data <- read.csv(file = url, header = FALSE,
                               col.names = c("patient.id", "CPTH", "CLSZ", "CLSP", "MADH", "EPTH", "BNUC", 
                                             "BLCR", "NNUC", "MTSE", "class"))
                               
head(breast_cancer_data)
tail(breast_cancer_data)
sum(breast_cancer_data$BNUC == "?") # For Missing Values
                               
                              
head(breast_cancer_data)
                               
breast_cancer_data <- breast_cancer_data[breast_cancer_data$BNUC != "?",] %>% mutate(BNUC = as.integer(as.character((BNUC))))
                               
sum(breast_cancer_data$BNUC == "?") # Check Null Values
                               
summary(breast_cancer_data)

#----------------------------------------------
# Feature Selection 

# ensure results are repeatable
set.seed(7)



# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(class~., data=breast_cancer_data, method="rf", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(breast_cancer_data[,1:10], breast_cancer_data[,11], sizes=c(1:10), rfeControl=control)

#results <- rfe(bcdata[,2:10], bcdata[,11], sizes=c(2:10), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
data <- predictors(results)
# plot the results

plot(results, type=c("g", "o"))


breast_cancer_data <- select(breast_cancer_data, -c(patient.id,MTSE))

head(breast_cancer_data)
#----------------------------------------------
# -------- EDA ----------

# normalization of dataset
# normalization of dataset using MinMax metod 


minval <- apply(breast_cancer_data[,1:9], 2, min) # for minimum value in the dataset
maxval <- apply(breast_cancer_data[,1:9], 2, max) # for maximum value in the dataset

scaledf <- scale(breast_cancer_data[,1:9], center=minval, scale=maxval-minval) # dataset scaling 


diagnosis <- as.factor(breast_cancer_data$class)

ggplot(breast_cancer_data, aes(x = diagnosis)) +
  geom_bar(fill = "#fc9272") +
  ggtitle("Distribution of Class in the Entire Dataset") +
  theme_minimal() +
  theme(legend.position = "none")



correlation <- cor(breast_cancer_data[,1: 9])
corrplot(correlation, type = "lower", col = c("#fcbba1", "#b2d2e8"), addCoef.col = "black", tl.col = "black") 

# proportion of 'benign' and 'malignant' class in dataset(breast_cancer_data)

round(prop.table(table(breast_cancer_data$class)),2)


bcdata <- data.frame(scaledf)  # convert dataset into dataFrame

head(bcdata) # head of the dataFrame



# traindata and testdata partioning with two indices

index <- sample(2, nrow(bcdata), replace=T, prob=c(0.7,0.3))

traindata <- bcdata[index==1,]
testdata <- bcdata[index==2,]

head(traindata)

# number of instance in traindata
nrow(traindata)

# number of 'benign' and 'malignant' class in traindata
table(traindata$class)

# proportion of 'benign' and 'malignant' class cancer cell
round(prop.table(table(traindata$class)),2)


head(testdata)

# number of instance in testdata
nrow(testdata)

# number of 'benign' and 'malignant' class in testdata
table(testdata$class)

# proportion of 'benign' and 'malignant' class cancer cell
round(prop.table(table(testdata$class)),2)


#----------------------------------------------

# Neural Network Model

# Fit traindata in NN Model;

nnfit <- nnet(class~., data=traindata, size=5, decay=1e-03, linout=F, 
              skip=F, maxit=1000, Hess=T)

summary(nnfit)


plotnet(nnfit, cex=0.7)

legend("bottomright", legend=c("CPTH  clump.thick", "CLSZ  cell.size", 
                               "CLSP  cell.shape", "MADH mar.adhesion", "EPTH  epithelial",
                               "BNUC  bare.nuclei", "BLCR  bland.chromatin", "NNUC normal.nuclei", 
                               "class   Tumur class"), bty="n", cex=0.6)


#----------------------------------------------

# fitted values for breastcancer with traindata in NN

fitval <- fitted.values(nnfit)
head(fitval)

fitval <- ifelse(fitval > 0.5, 1, 0)
result1 <- data.frame(classN = bcdata$class[index==1], fitval=fitval)
head(result1)


# confusion matrix for traindata for NN
tb1 <- table(result1)
tb1


# correct classification for traindata:
sum(diag(tb1))

# misclassification for traindata:
misclass1 <- sum(tb1) - sum(diag(tb1))
misclass1

# classification accuracy for traindata
accuracy1 = sum(diag(tb1))/sum(tb1)
accuracy1


# classification error for traindata
error1 <- 1-sum(diag(tb1))/sum(tb1)
error1


#----------------------------------------------

#----------------------------------------------

# Support Vector Machine 

# Fit traindata in SVM Model 
svmfit <- svm(class~., data = traindata, kernel = "linear", cost = 10, scale = FALSE)
summary(svmfit)


plot(svmfit)
#----------------------------------------------

# fitted values for breastcancer with traindata in SVM


svm_fitval <- fitted.values(svmfit)

head(svm_fitval)

svm_fitval <- ifelse(svm_fitval > 0.5, 1, 0)

svm_result <- data.frame(classN = bcdata$class[index==1], svm_fitval=svm_fitval)

head(svm_result)

# total number of instances in traindata
nrow(svm_result)

# confusion matrix for traindata
svm_tb1 <- table(svm_result)
svm_tb1


# correct classification for traindata:
sum(diag(svm_tb1))

# misclassification for traindata:
svm_misclass1 <- sum(svm_tb1) - sum(diag(svm_tb1))
svm_misclass1

# classification accuracy for traindata
svm_accuracy1 = sum(diag(svm_tb1))/sum(svm_tb1)
svm_accuracy1

# classification error for traindata
svm_error1 <- 1-sum(diag(svm_tb1))/sum(svm_tb1)
svm_error1


#----------------------------------------------


# predicted breastcancer malignant for testdata in NN

fpre <- predict(nnfit, newdata=testdata[,1:9])

predval <- fpre[,1]
head(predval)

predval <- ifelse(predval>0.5,1,0)
result2 <- data.frame(classN=bcdata$class[index==2], predval=predval)
head(result2)

# total number of instances in testdata
nrow(result2)

# confusion matrix for testdata
tb2 <- table(result2)
tb2

# correct classification for testdata:
sum(diag(tb2))

# misclassification for testdata:
misclass2 <- sum(tb2) - sum(diag(tb2))
misclass2

# classification accuracy for testdata
accuracy2 = sum(diag(tb2))/sum(tb2)
accuracy2

# classification error for testdata
error2 <- 1-sum(diag(tb2))/sum(tb2)
error2


#----------------------------------------------


# predicted breastcancer malignant for testdata in SVM

svm_pre <- predict(svmfit, newdata=testdata[,1:9])

svm_predval <- svm_pre

head(svm_predval)

svm_predval <- ifelse(svm_predval>0.5,1,0)

svm_result2 <- data.frame(classN=bcdata$class[index==2], svm_predval=svm_predval)

head(svm_result2)

# total number of instances in testdata
nrow(svm_result2)

# confusion matrix for testdata
svm_tb2 <- table(svm_result2)
svm_tb2

# correct classification for testdata:
sum(diag(svm_tb2))

# misclassification for testdata:
svm_misclass2 <- sum(svm_tb2) - sum(diag(svm_tb2))
svm_misclass2

# classification accuracy for testdata
svm_accuracy2 = sum(diag(svm_tb2))/sum(svm_tb2)
svm_accuracy2

# classification error for testdata
svm_error2 <- 1-sum(diag(svm_tb2))/sum(svm_tb2)
svm_error2




# matrix plot of the breast cancer cell data
pairs(breast_cancer_data[,1:9], col=breast_cancer_data[,9]+2, pch=(breast_cancer_data[,9]+16))

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------


plot(breast_cancer_data$CPTH, breast_cancer_data$CLSZ, col=breast_cancer_data[,9]+2, pch=(breast_cancer_data[,9]+16), 
     xlab = "clump thickness", ylab = "cell size") 

library(ggplot2)

ggplot(breast_cancer_data, aes(x=CPTH, y=CLSZ)) +
  geom_point(aes(colour=factor(class), shape=factor(class)), size=2) +
  xlab("clump thickness") +
  ylab("cell size") +
  ggtitle("clth vs clsz")

