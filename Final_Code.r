
#Load the packages

require("caTools")
require("ggplot2")
require("tidyr")
require("purrr")
require("reshape2")
require("devtools")
require("gclus")
require("ModelMetrics")
require("corrplot")
library("ROCR")


#Load the file

popularity <- read.csv("OnlineNewsPopularity.csv")
set.seed(100)


#Each column has correct data type

factorCols <- c("data_channel_is_lifestyle", "data_channel_is_bus", "data_channel_is_socmed", 
                "data_channel_is_tech","data_channel_is_world", "weekday_is_monday","weekday_is_tuesday",
                "weekday_is_wednesday","weekday_is_thursday","weekday_is_friday","weekday_is_saturday",
                "weekday_is_sunday","is_weekend","data_channel_is_entertainment") 
popularity[,factorCols] <- lapply(popularity[,factorCols], factor)



#Split data in 70% train set and 30% test set

popularity$url <- sample.split(popularity$url,0.7)
train <- subset(popularity, popularity$url == TRUE)
test <- subset(popularity, popularity$url == FALSE)
train <- train[,-c(1,2)]
test <- test[,-c(1,2)]


#Feature Scaling and Standardization

scaleCol <- c(1,2,6,7,8,10,11,18,19,20,21,22,23,24,25,26,27,28,29,59)
factorScale <- c(1,2,6,7,8,10,11,18,19,20,21,22,23,24,25,26,27,28,29,59,12,13,14,15,16,17,30,31,32,33,34,35,36,37)
trainscale <- train
testscale <- test
trainscale[,scaleCol] <- apply(train[,scaleCol], MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
trainscale[,-factorScale] <- apply(train[,-factorScale], MARGIN = 2, FUN = function(X)(X-mean(X))/sd(X)) 
testscale[,scaleCol] <- apply(test[,scaleCol], MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
testscale[,-factorScale] <- apply(test[,-factorScale], MARGIN = 2, FUN = function(X)(X-mean(X))/sd(X)) 


#Independent variables and dependent variables
                                  
x.train <- trainscale[,-59]
x.test <- testscale[,-59]
y.train <- trainscale[,59]
y.test <- testscale[,59]

#====================================
###Task1.Build linear regression
variable.correlation <- trainscale[,-c(12,13,14,15,16,17,30,31,32,33,34,35,36,37)]
variable.correlation <- cor(variable.correlation)
corrplot(variable.correlation, method="color", type="lower")
linearMod <- lm(shares ~ ., data=trainscale)


#Use MSE to evaluate the linear regression model
mse.lmtrain <- mse(y.train,linearMod$fitted.values)
y.test.pred <- predict(linearMod, x.test)
mse.lmtest <- mse(y.test,y.test.pred)
lm.mse <- as.data.frame(cbind(mse.lmtrain,mse.lmtest))


###Task2.Logistic regression
##Plot distribution of Share
ggplot(data=trainscale, aes(trainscale$shares)) + 
  geom_histogram(breaks=seq(0, 0.06, by= 0.001), 
                 fill="red", 
                 alpha = .8) + 
  labs(title="Histogram for Shares", x="Shares", y="Count")


#Generate class 1 and class 0 for shares 
##1 for large number of shares, 0 for small number of shares
trainLogit <- trainscale
testLogit <- testscale
trainLogit$shares <- ifelse(trainscale$shares>median(trainLogit$shares),1,0)
testLogit$shares <- ifelse(testscale$shares>median(testLogit$shares),1,0)
y.trainlogit <- trainLogit[,59]
y.testlogit <- testLogit[,59]

#=============================
#Build logistic regression
glm.model <- glm(shares ~.,family=binomial(link='logit'),data = trainLogit)
summary(glm.model)
y.trainlogit.pred <- glm.model$fitted.values
y.trainlogit.pred <- ifelse(y.trainlogit.pred > 0.5,1,0)
train.misClasificError <- mean(y.trainlogit.pred != y.trainlogit)
logit.train.accuracy <- (1-train.misClasificError)*100
y.testlogit.predict <- predict(glm.model,x.test, type = 'response')
y.testlogit.pred <- ifelse(y.testlogit.predict > 0.5,1,0)
test.misClasificError <- mean(y.testlogit.pred != y.testlogit)
logit.test.accuracy <- (1-test.misClasificError)*100
glm.accuray <- as.data.frame(cbind(logit.train.accuracy,logit.test.accuracy))

#ROC Plot
pr <- prediction(y.testlogit.pred, y.testlogit)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


#=========================
##Task2: Experiements
#Define Gradient Descent Algorithmn for linear regression
# define the gradient function dJ/dtheata: 1/m * (h(x)-y))*x where h(x) = x*theta
# in matrix form this is as follows:


# initialize coefficients
num.features <- ncol(x.train)
theta <- matrix(rep(0, num.features+1), nrow=1)


# add a column of 1's for the intercept coefficient

X<- data.matrix(cbind(1, x.train))
y <- y.train

# learning rate and iteration limit
num_iters <- 1000
alpha.values <- c( 0.001,0.003,0.01,0.03,0.1,0.3) 

# gradient descent
# keep history
cost_history <- double(num_iters)
theta_history <- list(num_iters)
cost.history <- double(length(alpha.values))
theta.history <- list(length(alpha.values))

#Experiment with various learning rate
for (j in 1:length(alpha.values)) {
  
  alpha = alpha.values[j]
  
  # squared error cost function
  cost <- function(x, y, theta) {
    sum( (x %*% t(theta) - y)^2 ) / (2*length(y))
  }
  
  #Gradient Descent
  for (i in 1:num_iters) {
    error <- (X %*% t(theta) - y)
    delta <- t(X) %*% error / length(y)
    theta <- theta - alpha * t(delta)
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
  }
  cost.history <- cbind(cost.history,cost_history)
  theta.history[[j]] <- theta_history
}


#Find the iteration for best theta value with each alpha values
cost.history <- as.data.frame(cost.history)
MSE.Train.location <- double(length(alpha.values))

MSE.Location.Matrix <- for(i in 2:(ncol(cost.history))){
  MSE.Train.location[i] <- which.min(cost.history[,i])
}


#Apply model on the test set
MES.Len<- length(MSE.Train.location)
MSE.Train.location <- as.data.frame(MSE.Train.location[2:MES.Len])
MSE.Test.Matrix <- double(length(alpha.values))

MSE.Test <- function(test, test.pred) {
  for (i in 1:(nrow(MSE.Train.location))) {
    t <- MSE.Train.location[i,1]
    Traintheta <- theta.history[[i]][[t]]
    test <- data.matrix(test)
    MSE.Test.Matrix[i] <- sum( (test %*% t(Traintheta) - test.pred)^2 ) / (2*length(test.pred))
  }
  return(MSE.Test.Matrix)
}

X.test <- data.matrix(cbind(1, x.test))
MSE.Test.output1 <- MSE.Test(X.test,y.test)
best.Alpha1 <- alpha.values[which.min(MSE.Test.output1)]
alpha.MSE<- as.data.frame(cbind(alpha.values,MSE.Test.output1))
alpha.values.no.six <- alpha.values[-6]
MSE.Test.output1.no.six <- MSE.Test.output1[-6]
MSE.Test.whole <- min(MSE.Test.output1)
dev.off()
qplot(alpha.values.no.six,MSE.Test.output1.no.six,geom=c("line"), xlab="Alpha Values", ylab="Test MSE")


#Plot the cost against number of iterations
cost.history <- as.data.frame(cost.history[,-1])
colnames(cost.history) <- c("cost1","cost2","cost3","cost4")
qplot(seq(1,nrow(cost.history),1), cost.history[,1], geom=c("line"), xlab="iteration", ylab="cost")
qplot(seq(1,nrow(cost.history),1), cost.history[,2], geom=c("line"), xlab="iteration", ylab="cost")
qplot(seq(1,nrow(cost.history),1), cost.history[,3], geom=c("line"), xlab="iteration", ylab="cost")
qplot(seq(1,nrow(cost.history),1), cost.history[,4], geom=c("line"), xlab="iteration", ylab="cost")
qplot(seq(1,nrow(cost.history),1), cost.history[,5], geom=c("line"), xlab="iteration", ylab="cost")


#================================================================
#Randomly choose 10 columns
ranCols <- sample(ncol(train[,-59]),10)
x.random  <- trainscale[,ranCols]
x.test.random <- testscale[,ranCols]


# initialize coefficients
num.features <- ncol(x.random)
theta <- matrix(rep(0, num.features+1), nrow=1)


# add a column of 1's for the intercept coefficient
X.random <- data.matrix(cbind(1,x.random))
X <- X.random

alpha.values <- 0.1
num_iters <- 1000

# gradient descent to retrain the model
# keep history
cost_history.random <- double(num_iters)
theta_history.random <- list(num_iters)

for (i in 1:num_iters) {
    error <- (X %*% t(theta) - y)
    delta <- t(X) %*% error / length(y)
    theta <- theta - alpha * t(delta)
    cost_history.random[i] <- cost(X, y, theta)
    theta_history.random[[i]] <- theta
  }

#Apply model on the test set
X.test.random <- data.matrix(cbind(1, x.test.random))
best.iteration <- which.min(cost_history.random)
MSE.Test.random <- sum((X.test.random %*% t(theta_history.random[[1000]]) - y.test)^2 ) / (2*length(x.test.random))


#Plot the cost against number of iterations
cost_random<- data.matrix(cost_history.random)
qplot(seq(1,nrow(cost_random),1), cost_random, geom=c("line"), xlab="iteration", ylab="cost")


#================================================================
#Pick 10 columns
pickCols <- c("num_imgs", "weekday_is_monday", "is_weekend", "abs_title_subjectivity",
              "abs_title_sentiment_polarity", "global_sentiment_polarity", "global_subjectivity",
              "max_positive_polarity", "max_negative_polarity", "average_token_length")

x.pick<- trainscale[,pickCols]
x.test.pick <- testscale[,pickCols]


# initialize coefficients
num.features <- ncol(x.random)
theta <- matrix(rep(0, num.features+1), nrow=1)


# add a column of 1's for the intercept coefficient
X.pick <- data.matrix(cbind(1,x.pick))
X <- X.pick

# gradient descent to retrain the model
# keep history
cost_history.pick <- double(num_iters)
theta_history.pick <- list(num_iters)

for (i in 1:num_iters) {
  error <- (X %*% t(theta) - y)
  delta <- t(X) %*% error / length(y)
  theta <- theta - alpha * t(delta)
  cost_history.pick[i] <- cost(X, y, theta)
  theta_history.pick[[i]] <- theta
}

#Apply model on the test set
X.test.pick <- data.matrix(cbind(1, x.test.pick))
best.iteration <- which.min(cost_history.pick)
MSE.Test.pick <- sum((X.test.pick %*% t(theta_history.pick[[1000]]) - y.test)^2 ) / (2*length(x.test.pick))


#Plot the cost against number of iterations
cost_pick<- data.matrix(cost_history.pick)
qplot(seq(1,nrow(cost_pick),1), cost_pick, geom=c("line"), xlab="iteration", ylab="cost")


#MSE Summary for Linear Regression
MSE<- rbind(MSE.Test.whole,MSE.Test.random,MSE.Test.pick)
test <- c("MSE for All", "MSE for Random 10","MSE for Pick 10")
combine.result.mse <- as.data.frame(cbind(test,MSE))
colnames(combine.result.mse) <- c("test","MSE")

ggplot() + geom_bar(aes(y = MSE, x = test),
                     data = combine.result.mse, stat="identity")


                                  
#====================================================================
##Task3: Logistic Regression
#Gradient Descent for logistic regression

# Implement Sigmoid function
sigmoid <- function(z) {
  g <- 1/(1+exp(-z))
  return(g)
}

#Cost function for logistic regression
cost.glm <- function(x, y, theta1){
  m = nrow(x)
  hx = sigmoid(x %*% t(theta1))
  return (1/m) * (((-t(y) %*% log(hx)) - t(1-y) %*% log(1 - hx)))
}


# Gradient descent function
gradLog <- function(x, y, theta1) {
  gradient <- (1 / nrow(y)) * (t(x) %*% (1/(1 + exp(-x %*% t(theta1))) - y))
  return(t(gradient))
}


gradientLog.descent <- function(x, y, threshold,alpha=0.001, num.iterations=1000,output.path=FALSE) {
  
  # Add x_0 = 1 as the first column
  m <- nrow(x)
  x <-  data.matrix(cbind(rep(1, m), x))
  
  num.features <- ncol(x)
  
  # Initialize the parameters
  theta1 <- matrix(runif(num.features), nrow=1)
  
  # Look at the values over each iteration
  theta.path <- theta1
  cost_history1 <- list()
  for (i in 1:num.iterations) {
    theta <- theta1 - alpha * gradLog(x, y, theta1)
    cost_history1[i] <- cost.glm(x, y, theta1)
    
    #print(cost.glm(x,y,theta))
    if(all(is.na(theta))) break
    theta.path <- rbind(theta.path, theta1)
    if(i > 2) if(all(abs(theta1 - theta.path[i-1,]) < threshold)) break 
  }
  
  if(output.path) return(theta.path) else return(theta.path[nrow(theta.path),])
}

threhold <- c(1e-7,1e-3)

#test 1
y.trainlogit <- data.matrix(y.trainlogit)
theta.logit1 <- gradientLog.descent(x= x.train, y = y.trainlogit, alpha = 0.2,num.iterations=3000, threshold = 1e-7,output.path=FALSE)
theta.logit.test1 <- data.matrix(theta.logit1)
X.test <- data.matrix(cbind(1, x.test))
y.testlogit.pred <-  sigmoid(X.test %*% theta.logit.test1)
table.threshold <- confusionMatrix(y.trainlogit,y.testlogit.pred,cutoff = 0.5)
accuracy.logit1 <- round(sum(table.threshold[1,1],table.threshold[2,2])/sum(table.threshold)*100,2)


#test 2
theta.logit2 <- gradientLog.descent(x=x.train, y = y.trainlogit, alpha = 0.2,num.iterations=3000, threshold = 1e-3,output.path=FALSE)
theta.logit2 <- data.matrix(theta.logit2)
Xtest <- data.matrix(cbind(1, x.test))
y.testlogit.pred1 <-  sigmoid(Xtest %*% theta.logit2)
table.threshold <- confusionMatrix(y.trainlogit,y.testlogit.pred1,cutoff = 0.5)
accuracy.logit2 <- round(sum(table.threshold[1,1],table.threshold[2,2])/sum(table.threshold)*100,2)

tablecool <- rbind(accuracy.logit1,accuracy.logit2)
Testcool <- c("Threshold 1", "Threshold 2")
tablecool <- cbind(Testcool,tablecool)
colnames(tablecool) <- c("test","accuracy")
#================================================================
#Randomly choose 10 columns

ranCols <- sample(ncol(train[,-59]),10)
x.random  <- trainscale[,ranCols]
x.test.random <- testscale[,ranCols]


theta.logit.random <- gradientLog.descent(x=x.random, y = y.trainlogit, alpha = 0.2,num.iterations=3000, threshold = 1e-7,output.path=FALSE)
theta.logit.random <- data.matrix(theta.logit.random)
Xtest <- data.matrix(cbind(1, x.test.random))
y.test.logit.pred2 <-  sigmoid(Xtest %*% theta.logit.random)
table.threshold1 <- confusionMatrix(y.trainlogit,y.test.logit.pred2,cutoff = 0.5)
accuracy2.1 <- round(sum(table.threshold1[1,1],table.threshold1[2,2])/sum(table.threshold1)*100,2)


#================================================================
#Pick 10 columns

pickCols <- c("num_imgs", "weekday_is_monday", "is_weekend", "abs_title_subjectivity",
              "abs_title_sentiment_polarity", "global_sentiment_polarity", "global_subjectivity",
              "max_positive_polarity", "max_negative_polarity", "average_token_length")

x.pick<- trainscale[,pickCols]
x.test.pick <- testscale[,pickCols]

theta.logit.pick <- gradientLog.descent(x=x.random, y = y.trainlogit, alpha = 0.2,num.iterations=3000, threshold = 1e-7,output.path=FALSE)
theta.logit.pick <- data.matrix(theta.logit.pick)
Xtest <- data.matrix(cbind(1, x.test.random))
y.test.logit.pred3 <-  sigmoid(Xtest %*% theta.logit.pick)
table.threshold2 <- confusionMatrix(y.trainlogit,y.test.logit.pred3,cutoff = 0.5)
accuracy3.1 <- round(sum(table.threshold2[1,1],table.threshold2[2,2])/sum(table.threshold2)*100,2)


#Accuracy Summary
logit.accuracy.whole <- as.data.frame(rbind(accuracy.logit2,accuracy2.1,accuracy3.1))
logist.test <- c("Whole","Random","Pick")
accuracy.logit.table <- as.data.frame(cbind(logist.test,logit.accuracy.whole))
ggplot() + geom_bar(aes(y = logit.accuracy.whole, x = logist.test),
                    data = accuracy.logit.table, stat="identity")

