# Cab Fare Prediction 
rm(list = ls())
setwd("D:/Python/Car Prediction/R code")

# loading datasets
train = read.csv(choose.files(), header = T)
test = read.csv(choose.files(), header = T)
# Structure of data
str(train)
str(test)
summary(train)
summary(test)
# converting the features in the required data types.
train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count=round(train$passenger_count)
########### data cleaning  #############
# fare amount cannot be less than one 
# considring fare amount 453 as max and removing all the fare amount greater than 453, as chances are
# very less of fare amount having 4000 and 5000 ...etc
train[which(train$fare_amount < 1 ),]
nrow(train[which(train$fare_amount < 1 ),]) # to show the count i.e.,5
train = train[-which(train$fare_amount < 1 ),]  # removing those values.
train[which(train$fare_amount>453),]
nrow(train[which(train$fare_amount >453 ),]) # to show the count i.e., 2
train = train[-which(train$fare_amount >453 ),]  # removing those values.
# passenger count cannot be Zero
# even if we consider suv max seat is 6, so removing passenger count greater than 6.
train[which(train$passenger_count < 1 ),]
nrow(train[which(train$passenger_count < 1 ),]) # to show count, that is 58
train=train[-which(train$passenger_count < 1 ),] # removing the values
train[which(train$passenger_count >6 ),]
nrow(train[which(train$passenger_count >6 ),]) # to show count, that is 20
train=train[-which(train$passenger_count >6 ),] # removing the values
# Latitudes range from -90 to 90.Longitudes range from -180 to 180.
# Removing which does not satisfy these ranges.
print(paste('pickup_longitude above 180=',nrow(train[which(train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train[which(train$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train[which(train$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train[which(train$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(train[which(train$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train[which(train$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train[which(train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train[which(train$dropoff_latitude > 90 ),])))
train = train[-which(train$pickup_latitude > 90),] # removing one data point
# Also we will see if there are any values equal to 0.
nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
# removing those data points.
train=train[-which(train$pickup_longitude == 0 ),]
train=train[-which(train$dropoff_longitude == 0),]
sum(is.na(train))
sum(is.na(test))
train=na.omit(train)
sum(is.na(train))  
# deriving the new features using pickup_datetime and coordinated provided.
# new features will be year,month,day_of_week,hour
# Convert pickup_datetime from factor to date time
train$pickup_datetime=as.Date(train$pickup_datetime)
pickup_time = strptime(train$pickup_datetime,format='%Y-%m-%d %H:%M:%S UTC')
train$date = as.integer(format(train$pickup_date,"%d"))# Monday = 1
train$mnth = as.integer(format(train$pickup_date,"%m"))
train$yr = as.integer(format(train$pickup_date,"%Y"))
#train$min = as.integer(format(train$pickup_date,"%M"))
#train$day=as.integer(as.POSIXct(train$pickup_datetime),abbreviate=F)
# for test data set.
test$pickup_datetime=as.Date(test$pickup_datetime)
pickup_time = strptime(test$pickup_datetime,format='%Y-%m-%d %H:%M:%S UTC')
test$date = as.integer(format(test$pickup_date,"%d"))# Monday = 1
test$mnth = as.integer(format(test$pickup_date,"%m"))
test$yr = as.integer(format(test$pickup_date,"%Y"))
#test$min = as.integer(format(test$pickup_date,"%M"))

# outlier
library(ggplot2)
pl1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)
# deriving the new feature, distance from the given coordinates.
library(geosphere)

train$dist= distHaversine(cbind(train$pickup_longitude, train$pickup_latitude), cbind(train$dropoff_longitude,train$dropoff_latitude))
#the output is in metres, Change it to kms
train$dist=as.numeric(train$dist)/1000
test$dist= distHaversine(cbind(test$pickup_longitude, test$pickup_latitude), cbind(test$dropoff_longitude,test$dropoff_latitude))
#the output is in metres, Change it to kms
test$dist=as.numeric(test$dist)/1000
# removing the features, which were used to create new features.
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,pickup_datetime))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,pickup_datetime))
str(train)
summary(train)
nrow(train[which(train$dist ==0 ),])
nrow(test[which(test$dist==0 ),])
nrow(train[which(train$dist >130 ),]) # considering the distance 130 as max and considering rest as outlier.
nrow(test[which(test$dist >130 ),])
# removing the data points by considering the above conditions,
train=train[-which(train$dist ==0 ),]
train=train[-which(train$dist >130 ),]
test=test[-which(test$dist ==0 ),]
# feature selection
numeric_index = sapply(train,is.numeric) #selecting only numeric
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data)
#Correlation analysis for numeric variables
library(corrgram)
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")
#removing date
# pickup_weekdat has p value greater than 0.05 
train = subset(train,select=-date)
#remove from test set
test = subset(test,select=-date)
library(car)
library(MASS)
qqPlot(train$fare_amount) # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(train$fare_amount) # truehist() scales the counts to give an estimate of the probability density.
lines(density(train$fare_amount)) # lines() and density() functions to overlay a density plot on histogram
d=density(train$fare_amount)
plot(d,main="distribution")
polygon(d,col="green",border="red")

D=density(train$dist)
plot(D,main="distribution")
polygon(D,col="green",border="red")

A=density(test$dist)
plot(A,main="distribution")
polygon(A,col="black",border="red")
#Normalisation
# log transformation.
train$fare_amount=log1p(train$fare_amount)
test$dist=log1p(test$dist)
train$dist=log1p(train$dist)
d=density(train$fare_amount)
plot(d,main="distribution")
polygon(d,col="green",border="red")
D=density(train$dist)
plot(D,main="distribution")
polygon(D,col="red",border="black")
A=density(test$dist)
plot(A,main="distribution")
polygon(A,col="black",border="red")
print('fare_amount')
train[,'fare_amount'] = (train[,'fare_amount'] - min(train[,'fare_amount']))/
(max(train[,'fare_amount'] - min(train[,'fare_amount'])))
train[,'dist'] = (train[,'dist'] - min(train[,'dist']))/
 (max(train[,'dist'] - min(train[,'dist'])))
test[,'dist'] = (test[,'dist'] - min(test[,'dist']))/
 (max(test[,'dist'] - min(test[,'dist'])))
###check multicollearity
library(usdm)
vif(train[,-1])
vifcor(train[,-1], th = 0.9)
#No variable from the 4 input variables has collinearity problem. 
#The linear correlation coefficients ranges between: 
#min correlation ( mnth ~ passenger_count ):  -0.001868147 
#max correlation ( yr ~ mnth ):  -0.1091115 

# ---------- VIFs of the remained variables -------- 
#   Variables      VIF
# 1 passenger_count 1.000583
# 2            mnth 1.012072
# 3              yr 1.012184
# 4        distance 1.000681

## to make sure that we dont have any missing values
sum(is.na(train))
train=na.omit(train)
# model building
#create sampling and divide data into train and test
set.seed(123)
train_index = sample(1:nrow(train), 0.8 * nrow(train))

train1 = train[train_index,]#do not add column if already removed
test1 = train[-train_index,]#do not add column if already removed
#################Decision Tree#####################################
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y*100))
}
library(rpart)
fit = rpart(fare_amount ~. , data = train1, method = "anova", minsplit=5)

summary(fit)
predictions_DT = predict(fit, test1[,-1])

MAPE(test1[,1], predictions_DT)

write.csv(predictions_DT, "DT_R_PRed8.csv", row.names = F)

#Error  12.94
#Accuracy 87.06

# Rndom forest
library(randomForest)
RF_model = randomForest(fare_amount ~.  , train1, importance = TRUE, ntree=100)
RF_Predictions = predict(RF_model, test1[,-1])

MAPE(test1[,1], RF_Predictions)

#error 12.69 for n=100
#accuracy = 87.31
# Linear model####
lm_model = lm(fare_amount ~. , data = train1)
summary(lm_model)

predictions_LR = predict(lm_model, test1[,-1])
MAPE(test1[,1], predictions_LR)

#error 13.55789
#Accuracy 86.44211

###Predict test data using random forest model

# Create the target variable
test$fare_amount=0
test= test[,c(1,2,3,4,5)]

RF_Predictions = predict(RF_model, test1[,-1])
RF_test=predict(RF_model, test[,-5])
write.csv(RF_test, "RF_R_PRed8.csv", row.names = F)






