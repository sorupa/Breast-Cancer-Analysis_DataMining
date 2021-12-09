# breast Cancer disease classification with logistic regression & decision tree:

library(ggplot2)
library(lattice)
library(ROCR)
library(corrplot)
library(dplyr)
library(PerformanceAnalytics)
library(psych)
library(corrgram)
library(tree)


# url for Breast Cancer data from Wisconsin

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

#Read the dataset and change the column headings
# id       Sample code number
# CT       Clump Thickness              UCSize  Uniformity of Cell Size 
# UCShape  Uniformity of Cell Shape     MA      Marginal Adhesion
# SECS     Single Epithelial Cell Size  BN      Bare Nuclei
# BC       Bland Chromatin              NN      Normal Nucleoli 
# M        Mitoses                      diagnosis benign/malignant
breast_cancer_data <- read.csv(file = url, header = FALSE,
                 col.names = c("id","CT", "UCSize", "UCShape", "MA", "SECS", "BN", "BC", "NN","M", "diagnosis") )

breast_cancer_data$outcome[breast_cancer_data$diagnosis==4] = 1
breast_cancer_data$outcome[breast_cancer_data$diagnosis==2] = 0
breast_cancer_data$outcome = as.integer(breast_cancer_data$outcome)
head(breast_cancer_data)

# Remove id column and ? values
data2 <- breast_cancer_data %>% select(-id, -BN)
data2$outcome[data2$diagnosis==4] = 1
data2$outcome[data2$diagnosis==2] = 0
data2$outcome = as.integer(data2$outcome)
head(data2)

# Create a Correlation Chart for Means
chart.Correlation(data2[, c(1:7)], histogram=TRUE, col="grey10", pch=1, main="Cancer Means")

# Create Correlation Chart for SE
pairs.panels(data2[,c(1:6)], method="pearson",
             hist.col = "#1fbbfa", density=TRUE, ellipses=TRUE, show.points = TRUE,
             pch=1, lm=TRUE, cex.cor=1, smoother=F, stars = T, main="SE")

# Split the dataset into the training sample and the testing sample using a 50/50 split
sample_size = floor(0.5 * nrow(data2))

# set the seed to make your partition reproductible
set.seed(1729)
train_set = sample(seq_len(nrow(data2)), size = sample_size)

training = data2[train_set, ]

testing = data2[-train_set, ]

head(training)
head(testing)

# Plotting diagnosis to see the distribution
ggplot(data2, aes(x = diagnosis)) +
  geom_bar(fill ="green") +
  ggtitle("Distribution of diagnosis for the entire dataset") +
  theme(legend.position="none")

ggplot(training, aes(x = diagnosis)) + 
  geom_bar(fill = 'blue') + 
  ggtitle("Distribution of diagnosis for the training dataset") + 
  theme(legend.position="none")

ggplot(testing, aes(x = diagnosis)) + 
  geom_bar(fill = 'orange') + 
  ggtitle("Distribution of diagnosis for the testing dataset") + 
  theme(legend.position="none")

# Corrgram of the entire dataset
corrgram(data2, order=NULL, lower.panel=panel.shade, upper.panel=NULL, text.panel=panel.txt,
         main="Corrgram of the data")

# Corrgram of the training dataset
corrgram(training, order=NULL, lower.panel=panel.shade, upper.panel=NULL, text.panel=panel.txt,
         main="Corrgram of the training data")

# Corrgram of the testing dataset
corrgram(testing, order=NULL, lower.panel=panel.shade, upper.panel=NULL, text.panel=panel.txt,
         main="Corrgram of the testing data")

# Model Fitting
# Start off with this (alpha = 0.05)
model_algorithm = model = glm(outcome ~ CT + 
                                UCSize +
                                UCShape +
                                MA +
                                SECS + 
                                BC  +
                                NN  +
                                M ,
                              family=binomial(link='logit'), control = list(maxit = 50),data=training)

print(summary(model_algorithm))

# Using Uniform Cell size and Uniform Cell Shape as predictors of diagnosis
# Settled Uniform Cell Size and Uniform Cell Shape
model_algorithm_final = model = glm(outcome ~ UCSize + UCShape ,
                                    family=binomial(link='logit'), control = list(maxit = 50),data=training)

print(summary(model_algorithm_final))

model_algorithm_final = model = glm(outcome ~ UCSize + UCShape + MA ,
                                    family=binomial(link='logit'), control = list(maxit = 50),data=training)

print(summary(model_algorithm_final))

# Apply the algorith to the training sample
prediction_training = predict(model_algorithm_final,training, type = "response")
prediction_training = ifelse(prediction_training > 0.5, 1, 0)
error = mean(prediction_training != training$outcome)
print(paste('Model Accuracy',1-error))

# Calcualte the ROC curve and the AUC 
# Get the ROC curve and the AUC
p = predict(model_algorithm_final, training, type="response")
pr = prediction(p, training$outcome)
prf = performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc = performance(pr, measure = "auc")
auc = auc@y.values[[1]]
print(paste("Model Accuracy", auc))

# Apply the algorithm to the testing sample
prediction_testing = predict(model_algorithm_final,testing, type = "response")
prediction_testing = ifelse(prediction_testing > 0.5, 1, 0)
error = mean(prediction_testing != testing$outcome)
print(paste('Model Accuracy',1-error))

# Get the ROC curve and the AUC
p = predict(model_algorithm_final, testing, type="response")
pr = prediction(p, testing$outcome)
prf = performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc = performance(pr, measure = "auc")
auc = auc@y.values[[1]]
print(paste("Model Accuracy", auc))

# Apply the algorithm to the entire dataset
prediction_data = predict(model_algorithm_final,breast_cancer_data, type = "response")
prediction_data = ifelse(prediction_data > 0.5, 1, 0)
error = mean(prediction_data != breast_cancer_data$outcome)
print(paste('Model Accuracy',1-error))

# Get the ROC curve and the AUC
p = predict(model_algorithm_final, breast_cancer_data, type="response")
pr = prediction(p, breast_cancer_data$outcome)
prf = performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc = performance(pr, measure = "auc")
auc = auc@y.values[[1]]
print(paste("Model Accuracy", auc))

# Decision Tree
# Droping the outcome variable which was used for the logistic model
training$outcome = NULL
testing$outcome = NULL

training$diagnosis[training$diagnosis == 4] = 1
training$diagnosis[training$diagnosis ==2] = 0

# Running our first tree 
model_tree = tree(diagnosis ~ UCSize + 
                    UCShape +
                    MA +
                    SECS +
                    BC  + 
                    NN  +
                    M,
                  data = training)

summary(model_tree)

# Now we want to plot our results
plot(model_tree, type = "uniform")

# Add some text to the plot
text(model_tree, pretty = 0, cex=0.8)
