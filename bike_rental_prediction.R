

library(lubridate)
library(caret)
library(randomForest)
library(Metrics)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ggplot2)






train <- read.csv('.../train.csv', header = T, sep = ',', dec = '.', stringsAsFactors = F )
test <- read.csv('.../test.csv', header = T, sep = ',', dec = '.', stringsAsFactors = F)




test$registered <- 0
test$casual <- 0
test$count <- 0

data <- rbind(train, test)

#sprawdzenie braków danych
table(is.na(data))

#Sprawdzenie rozkładów poszczególnych zmiennych i wytypowanie zmiennych do zmianych na factor
par(mfrow=c(4,2))
par(mar=rep(2,4))
hist(data$season)
hist(data$weather)
hist(data$humidity)
hist(data$holiday)
hist(data$workingday)
hist(data$temp)
hist(data$atemp)
hist(data$windspeed)

str(data)

# zamiana typu zmiennych na factor
to_factor <- c('season','weather','holiday','workingday')

data[,to_factor] <- data.frame(apply(data[to_factor],2, as.factor))
str(data)


# dodanie dodatkowych zmiennych czasu

data$datetime <- ymd_hms(data$datetime)
data$hour <- hour(data$datetime)
data$day <- wday(data$datetime)
data$year <- year(data$datetime)
data$month <- month(data$datetime)

#zmiana dodatkowych zmiennych na typ factor
data[,c('hour','day','year','month')] <- data.frame(apply(data[c('hour','day','year','month')],2, as.factor))



#wyodrębnienie zbioru z wartościami zmiennej count 
#(usunięcie zmiennych casual i registered, które są bezpośrednio skorelowane z ilością wypożyczeń
#a następnie podział na zbiór trennigowy i validacyjny

data$registered <- NULL
data$casual <- NULL

train <- data[(1:10886),]

index <- createDataPartition(train$count, p = .8, list = F)
t <- train[index,]
v <- train[-index,]

rf <- randomForest(count~., data=t, ntree=200)

rf

pred_train <- predict(rf, t)
pred_test <- predict(rf, v)

# Bardzo duża różnica w błędach, co świadczy o przetrenowaniu modelu
mape(t$count, pred_train)
#[1] 0.3469261
mape(v$count, pred_test)
#[1] 0.711283

###############
#Stworzenie dodatkowych zmiennych i sprawdzenie czy poprawią jakość modelu

#Drzewo decyzyjne dla zmiennej "hour"
par(mfrow=c(1,1))

hour_rpart <- rpart(count~hour,data=train)
rpart.plot(hour_rpart)

# wykres ilość wypożyczeń w poszczególnych godzinach
ggplot(train, aes(hour,count))+ geom_point()

data$hour <- as.integer(data$hour)


# Tworzenie zmiennej Day_part na podstawie wyników drzewa decyzyjnego

data$day_part <- 0
data$day_part[data$hour<8]<- 1
data$day_part[data$hour==8]<- 2
data$day_part[data$hour==9]<- 3
data$day_part[data$hour>9 & data$hour<=16]<- 4
data$day_part[data$hour==17]<-5
data$day_part[data$hour==18|data$hour==19]<-6
data$day_part[data$hour==21|data$hour==20] <- 7
data$day_part[data$hour>=22] <- 8



#Tworzenie zmiennj z przedziałami temperatur
#(Temp_part) na podstawie wyników drzewa decyzyjnego

temp_rpart <- rpart(count~temp, data=train)
rpart.plot(temp_rpart)

#wykres ilości wypożyczeń przy określonej temperaturze
ggplot(train, aes(temp,count))+ geom_point()



data$temp_part <- 0
data$temp_part[data$temp<13] <- 1
data$temp_part[data$temp>13 & data$temp<23] <- 2
data$temp_part[data$temp>=23 & data$temp<30] <- 3
data$temp_part[data$temp>=30] <- 4


# Zmienna określająca typ dnia, wyróżnienie świąt, weekendów i dni roboczych

data$day_type <- 0
data$day_type[data$holiday==0 & data$workingday==0] <- "1"
data$day_type[data$holiday==1] <- "2"
data$day_type[data$holiday==0 & data$workingday==1] <- "3"

# weekend =1
# holiday = 2
# workingday = 3

ggplot(data[1:10886,], aes(day_type,count))+ geom_point()

# zamiana typu zmiennych na factor
to_factor <- c('hour', 'day_part','temp_part', 'day_type','month','year')


data[,to_factor] <- data.frame(apply(data[to_factor], 2, as.factor))
str(data)


#### przypisanie braków danych zmiennej count na 
# zbiorze w którym te wartości nie były znane
data2 <- data
data2[-(1:10886),]$count <- NA

#posortowanie zbiru po dacie
data2 <- data2[order(data2$datetime),]

# dodanie zmiennych z wartościami zmiennej count opóźnionymi odpowiednio
# o 1 godzinę, 1 dzień, 7 dni
data2$lag1 <- lag(data2$count,1)
data2$lag24 <- lag(data2$count, 24)
data2$lag168 <- lag(data2$count, 168)

#pominięcie wartości z brakami danych
train <- na.omit(data2)


#zbiór test nie posiada wartości zmiennej count ,więc zbiór trenningowy i walidacyjny tworzymy ze zbioru train  
index <- createDataPartition(train$count, p = .8, list = F)
t <- train[index,]
v <- train[-index,]


# Ustalenie listy predyktorów
predictors<- colnames(train[,-which(names(train)=='count')])
# Zmienna celu
outcomeName<-'count'


# Tunning regresji liniowej z kroswalidacją i doborem parametrów regularyzacji

control <- trainControl(method = 'cv', number = 5)

cv_glmnet <- train(count ~ ., 
                   data = train, 
                   method = "glmnet", 
                   metric = 'RMSE', 
                   tuneGrid = expand.grid(alpha = seq(0.1,1,by = 0.1),lambda = seq(0.001,0.1,by = 0.001)),
                   trControl = control)

print(cv_glmnet)

# Ostateczny model funkcji regresji liniowej
coef(cv_glmnet$finalModel, cv_glmnet$bestTune$lambda)


pred_train <- predict(cv_glmnet, t)
pred_test <- predict(cv_glmnet, v)

# Po sprawdzeniu wyników na zbiorze treningowym i walidacyjnym różnica
# w błędach jest niewielka co świedczy o braku przetrenowania modelu
mape(t$count, pred_train)
#[1] 0.7927089
mape(v$count, pred_test)
#[1] 0.6503997



# Kroswalidacja drzewa decyzyjnego
CV_rpart <- train(count~.,
                  data=train,
                  method = "rpart2",
                  trControl=control,
                  tuneLength=19)

CV_rpart

pred_train <- predict(CV_rpart, t)
pred_test <- predict(CV_rpart, v)

# Po sprawdzeniu wyników na zbiorze treningowym i walidacyjnym różnica
# w błędach jest niewielka co świedczy o braku przetrenowania modelu
mape(t$count, pred_train)
#[1] 1.743365
mape(v$count, pred_test)
#[1] 1.629669


# włączenie obliczeń równoległych w celu przyspieszenia 

library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

#kroswalidacja i tunning modelu SVM

control <- trainControl(method = 'cv', number = 5, allowParallel = TRUE)

cv_svm <- train(count ~ .,
                data = train, 
                method = "svmLinear",
                metric = 'RMSE',
                trControl = control,
                preProcess = c("center", "scale"),
                tuneLength = 19)




cv_svm


pred_train <- predict(cv_svm, t)
pred_test <- predict(cv_svm, v)

# Po sprawdzeniu wyników na zbiorze treningowym i walidacyjnym różnica
# w błędach jest niewielka co świedczy o braku przetrenowania modelu
mape(t$count, pred_train)
#[1] 0.6997471
mape(v$count, pred_test)
#[1] 0.6028018



# Kroswalidacja i tuning parametrów lasów losowych 

control <- trainControl(method = 'cv', number = 5, allowParallel = TRUE)



cv_las <- train(count ~ ., 
                data = train, 
                method = "ranger", 
                trControl = control,
                importance = "impurity",
                num.trees = 500,
                preProcess = c("center", "scale", "BoxCox"),
                verbose=T
                )



cv_las


pred_train <- predict(cv_las, t)
pred_test <- predict(cv_las, v)

# Po sprawdzeniu wyników na zbiorze treningowym i walidacyjnym różnica
# w błędach jest niewielka co świedczy o braku przetrenowania modelu
mape(t$count, pred_train)
#[1] 0.1382172
mape(v$count, pred_test)
#[1] 0.1332671


stopCluster(cluster)


#save(cv_las, cv_svm, file = "RF_svm_bike.rda")

#load('RF_svm_bike.rda')


#porównanie modeli

resamps <- resamples(list(GLMNET = cv_glmnet,
                          RPART = CV_rpart,
                          RF = cv_las,
                          SVM = cv_svm))
summary(resamps)
trellis.par.set(caretTheme())
dotplot(resamps, metric = "RMSE")


pred <- data2
model <- cv_las

#uzupełnienie wartości zmiennej count wartościami predykcji modelu cv_las
for (i in 1:nrow(pred)) {
  if (is.na(pred[i,'count'])) {
    pred[i,'count'] <- predict(model, pred[i,])
    
    pred$lag1 <- lag(pred$count,1)
    pred$lag24 <- lag(pred$count, 24)
    pred$lag168 <- lag(pred$count, 168)
    
  }
}


#wykres prównania wartości rzeczywistych z prognozowanymi
wykres <- cbind(train[,c('datetime','count')], predict=predict(cv_las, train))
  
ggplot(wykres, aes(datetime, count)) +
  geom_point(na.rm=TRUE, color="blue", size=1) + 
  ggtitle("Porównanie prognozy z wartościami rzeczywistymi") +
  xlab("Datetime") + ylab("Count")+
  geom_line(aes(y=predict), color='black')


# #wykres prównania wartości rzeczywistych z prognozowanymi dla jedneo miesiąca
ggplot(wykres[wykres$datetime>'2012-07-01 00:00:00' & wykres$datetime<='2012-07-31',], aes(datetime, count)) +
  geom_point(na.rm=TRUE, color="blue", size=1) + 
  ggtitle("Porównanie prognozy z wartościami rzeczywistymi dla lipca 2012") +
  xlab("Datetime") + ylab("Count")+
  geom_line(aes(y=predict), color='black')



# wykres wartości rzeczywistych zbioru train i prognozy  przygotowanej dla zbioru test

prognoza <- pred[day(pred$datetime)>=20,c('datetime','count')]
colnames(prognoza) <- c('datetime','prediction')
wykres2 <- merge(x = data2,  y= prognoza, by = "datetime", all = TRUE)

ggplot(wykres2, aes(datetime, count)) +
  geom_line(na.rm=TRUE, color="black", size=1) + 
  ggtitle("Wartości rzeczywiste i prognoza dla zbioru test") +
  xlab("Datetime") + ylab("Count")+
  geom_line(aes(y=prediction), color='blue')


# wykres dla jednego miesiąca
ggplot(wykres2[wykres2$datetime>'2012-07-01 00:00:00' & wykres2$datetime<='2012-07-31',], aes(datetime, count)) +
  geom_line(na.rm=TRUE, color="black", size=1) + 
  ggtitle("Wartości rzeczywiste i prognoza dla zbioru test dla lipca 2012") +
  xlab("Datetime") + ylab("Count")+
  geom_line(aes(y=prediction), color='blue')



















