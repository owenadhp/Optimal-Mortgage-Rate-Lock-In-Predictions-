#Installing necessary packages
#install.packages("FNN")
#install.packages("klaR")
#install.packages("caret")
#install.packages("class")


#Load all of the Library
library("klaR")
library("caret")
library("ggplot2")
library("scatterplot3d")
library("dplyr")
library("FNN")
library("class")

#initializing Data
rm(list=ls())
Data_full=read.csv("VIXSP500.csv")
head(Data_full)
summary(Data_full)

#Removing the NA from the Data
Data = Data_full[complete.cases(Data_full), ]


#Plot the data
plot(Data$VIX, Data$Log.returns,pch=20,col="blue",cex=1, main='Data and Fitted values')
plot(Data$Returns, Data$Log.returns,pch=20,col="blue",cex=1, xlim=c(-1,0.65), ylim=c(-0.05,0.05), main='Data and Fitted values')
plot(Data$Date, Data$Log.returns,pch=20,col="blue",cex=1, ylim=c(-0.05,0.05), main='Data and Fitted values')

#All scatterplots and Correlation
pairs(~Returns + VIX + SP500 + Log.returns, Data, upper.panel=NULL)
cor(data.frame(Data$Returns, Data$VIX, Data$SP500, Data$Log.returns ))


##Splitting the data using Caret package##
set.seed(98)
Data_Split = createDataPartition(Data$Date, p=0.75, list=FALSE)
train = Data[Data_Split,]
test = Data[-Data_Split,]

#Training Linear Model (Train) for Log.returns
Returns = train$Returns
Log.returns = train$Log.returns
VIX.ls = lm(Log.returns ~ Returns, train)
VIX.ls
summary(VIX.ls)
anova(VIX.ls)
beta0= VIX.ls$coefficients[1]
beta1= VIX.ls$coefficients[2]

Meanloss=function(beta,x,y){
  yhat=beta[1]+beta[2]*x
  L=mean((yhat-y)^2)
  return(L)
}
beta_M=optim(c(beta0,beta1),Meanloss, x=Returns, y=Log.returns)
plot(Returns,Log.returns,xlab='VIX.Returns', main='Returns_normalize and Log.returns', pch=20, col="blue", cex=1)
abline(a=beta_M$par[1],b=beta_M$par[2], col="red", cex=1.1)
abline(a=beta0,b=beta1, col="green", cex=1.1)

ggplot(data = lm_test_data) +
  geom_point(aes(x = Returns, y = Log.returns, colour = "blue")) +
  geom_line(aes(x = Returns, y = log_returns_predicted, colour = "green"), size = 1.5) +
  labs(x = "VIX Returns", y = "S&P500 LogReturns",
       title = "S&P500 Actual Returns vs. Predicted Returns") +
  scale_colour_discrete(name = "Data",
                        labels = c("Actual", "Predicted")) +
  theme(legend.position = "right",
        plot.title = element_text(size = 14, face = 'bold', family = "Arial", hjust = 0.5),
        axis.title.x = element_text(family = "Arial", size = 12),
        axis.title.y = element_text(family = "Arial", size = 12),
        text = element_text(family = "Arial", size = 10)
  )


#Testing Linear Model (Test) for Log.returns
y_hat  = beta0 + beta1 * test$Returns
y_hat
lm_test_data = mutate(test, log_returns_predicted = y_hat)


#Calculate the Mean Squared Error
MSE = sum((lm_test_data$Log.returns - lm_test_data$log_returns_predicted)^2)/(dim(test)[1]-1)
MSE * 100


#Calculate Residuals and see if the residuals are normal

residuals  = lm_test_data$Log.returns - lm_test_data$log_returns_predicted
plot(residuals, col = "black", lwd =1)
abline(a=0,b=0,col="red",cex=5. , lwd =2)

qqnorm(residuals)
qqline(residuals, col = "green", lwd = 2)



#########################
#######KNN Model#########


###Train our model using guess kNN###
Train_X=train$Returns
Train_Y=train$Log.returns
y_hat=c()
x=Train_X
k=31

for(i in 1:length(Train_X)){
  DD=sqrt((Train_X-x[i])^2)
  S=sort(DD, index.return=TRUE)
  I=S$ix[1:k]
  y_hat[i]=mean(Train_Y[I])
}
MSE_Train=mean((y_hat-Train_Y)^2)
MSE_Train*100
plot(Train_X,Train_Y,pch=20,col="blue",cex=0.8)
par(new=TRUE)
plot(Train_X,y_hat,pch=20,col="red",cex=0.8, main="Predicted values on the Train")



 
#Test/Validation
Test_X = test$Returns
Test_Y = test$Log.returns
y_hat_test=c()
x=Test_X
for(i in 1:length(Test_X)){
  DD=sqrt((Train_X-x[i])^2)
  S=sort(DD, index.return=TRUE)
  I=S$ix[1:k]
  y_hat_test[i]=mean(Train_Y[I])
}
MSE_Test=mean((y_hat_test-Test_Y)^2)
MSE_Test*100
plot(Test_X,Test_Y,pch=20,col="blue",cex=1,main="Predicted values on the Validation")
par(new=TRUE)
plot(Test_X,y_hat_test,pch=20,col="red",cex=1, xlab="", ylab="")


### Validation MSE in function of k
k_vector=seq(2, 80,1)
Test_X = test$Returns
Test_Y = test$Log.returns
x=Test_X
MSE_Test=c()
for(kk in 1:length(k_vector)){
  y_hat_test=c()
  k=k_vector[kk]
  for(i in 1:length(Test_X)){
    DD=sqrt((Train_X-x[i])^2)
    S=sort(DD, index.return=TRUE)
    I=S$ix[1:k]
    y_hat_test[i]=mean(Train_Y[I])
  }
  MSE_Test[kk]=mean((y_hat_test-Test_Y)^2)*100
}
plot(k_vector,MSE_Test, pch=20, col="blue",cex=1.2,xlab="k", ylab="MSE", main="All Possible MSE")




#Which MSE is the best
MSE_target = cbind(k_vector, MSE_Test)
best_k = MSE_target[10:40,]
best_k = best_k[which(best_k[,2] == min(best_k[,2])),]
MSE_KNN = best_k[2]
K_KNN = best_k[1]
K_KNN
MSE_KNN

######### Last Comparison of Linear Model and KNN MSE ########
MSE - MSE_KNN
#MSE_KNN bigger than MSE for LM, which mean MSE for LM is better at prediction compare to MSE_KNN 


###Can we predict SP500 up/down using VIX_returns?
lm_test_data = lm_test_data %>% mutate(direction = sign(Log.returns), direction_predicted = sign(log_returns_predicted)) %>%
  mutate(prediction_accurate = case_when(direction_predicted == direction ~ 1, direction_predicted != direction ~ 0))
lm_test_data$direction[lm_test_data$direction == 0] = 1

#Draw the Matrix to create table

direction_matrix = as.factor(lm_test_data$direction)
direction_lm_predicted = as.factor(lm_test_data$direction_predicted)
table(direction_lm_predicted, direction_matrix)


#Error Rate
lm_direction_error = 1-(sum(lm_test_data$prediction_accurate)/nrow(lm_test_data))
print(lm_direction_error)


