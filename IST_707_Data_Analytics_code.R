# ist707 final project
# authors
  # Cassie Alter
  # Benjamin Cowan
  # Katherine Gilleran


fname <- file.choose()
aus <- read.csv(fname, stringsAsFactors = FALSE)
aus <- na.omit(aus)

str(aus)
View(aus)


######################################################

# How many days out can we predict?

######################################################

# convert date to allow math to be done on it
conversion <- "%m/%d/%y"
aus$Date <- as.POSIXct(strptime(aus$Date, format = conversion))
aus_plus$Date <- as.POSIXct(strptime(aus_plus$Date, format = conversion))
# a day in seconds
day <- 86400


aus_plus$Rain_7days <- NA
for (i in 1:length(aus_plus$Date)) {
  # Rain value of same location and 5 days in future
  tmp <- aus_plus$RainToday[which(!is.na(match(aus_plus$Date, 
                                               (aus_plus$Date[i] + (day*7))) 
                                         & match(aus_plus$Location, 
                                                 aus_plus$Location[i])))]
  aus_plus$Rain_7days[i] <- if(identical(tmp, character(0)) == FALSE) {
    tmp
  } else {NA}
}


######################################################

# Which location can we predict best?

######################################################

# create data frame for each location
locations <- unique(aus$Location)

for (i in 1:length(locations)) {
  tmp <- aus1[which(aus1$Location == locations[i]), ]
  assign(paste0(locations[i]), tmp)
}


######################################################

# EDA

######################################################


# boxplots of Average Temperature by Location
library(ggplot2)
library(plyr)


# count of rain to non-rain
ggplot(data = aus, 
       aes(x = RainToday)) + 
  geom_bar(fill= c("lightblue2", "steelblue4")) +
  labs(title = "How many times did it rain?", 
       x = "Did it rain?")

# bar plot of rain tomorrow by rain today
ggplot(data = aus, 
       aes(x = RainToday, 
           fill = factor(RainTomorrow))) +
  geom_bar(position = "fill") +
  labs(title = "Predicting Rain by Rain", 
       x = "Did it rain today?",
       fill = "Will it rain tomorrow?") 

aus$Average_Temp <- (aus$MinTemp + aus$MaxTemp)/2
ggplot(data = aus, 
       aes(x = reorder(Location, Average_Temp, FUN = median), 
           y = Average_Temp)) + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, 
                                   vjust = 0.5, 
                                   hjust=1)) +
  labs(title = "Average Temperature in Australia", 
       x = "Location",
       y = "Average Temperature")


location_rain <- data.frame(table(aus$Location, aus$RainToday))
colnames(location_rain) <- c("Location", "Rain", "Freq")

location_rain <- location_rain[order(location_rain$Location), ]
library(tidyverse)
location_rain <- location_rain %>% arrange(desc(location_rain$Location))

ggplot(data=location_rain, 
       aes(x = Location, 
           y = Freq, 
           fill = Rain)) +
  geom_bar(stat="identity", position = "fill") +
  labs(title = "Rain by Location",
       y = "Percent Rain") + 
  coord_flip()



#heatmap
library(dplyr)
aus.num <- select_if(aus, is.numeric)
aus.char <- select_if(aus, is.character)
aus.num.cor <- cor(aus.num)

library(RColorBrewer)
palette = colorRampPalette(brewer.pal(8, 'Oranges'))(25)
heatmap(x=aus.num.cor, col=palette, symm = TRUE
        , margins = c(10,10))
legend(x="bottomright", legend=c("low", "moderate", "high"), 
       fill=colorRampPalette(brewer.pal(8, "Oranges"))(3))

#barplot
library(ggplot2)
aus %>%
  ggplot(aes(x= reorder(Location, MaxTemp), MaxTemp))+ 
  stat_summary(fun.y=mean, geom="bar", fill = "#6779AD")+
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))+
  coord_flip()

#boxplots
aus$avg_winds <- (aus$WindSpeed9am+aus$WindSpeed3pm)/2
aus %>%
  ggplot(aes(x= reorder(Location, avg_winds), avg_winds))+ 
  geom_boxplot(fill="#FFAE6B")+
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))+
  coord_flip()

aus$avg_hum <- (aus$Humidity9am+aus$Humidity3pm)/2
aus %>%
  ggplot(aes(x= reorder(Location, avg_hum), avg_hum))+ 
  geom_boxplot(fill="#EA8099")+
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))+
  coord_flip()

aus$avg_press <- (aus$Pressure9am+aus$Pressure3pm)/2
aus %>%
  ggplot(aes(x= reorder(Location, avg_press), avg_press))+ 
  geom_boxplot(fill="#A0DF7A")+
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))+
  coord_flip()



#practice
#par(las=1, mar = c(7,7,7,7))
#m <- tapply(aus$MaxTemp, aus$Location, mean)
#str(m)
#color stuff
#library(RColorBrewer)
#par(mar=c(3,4,2,2))
#display.brewer.all()
#my_pal <- colorRampPalette(brewer.pal(8, "Oranges"))(3)


#associative rules
tweettrans_rules = arules::apriori(aus1
                                   ,parameter = list(support=.06, confidence=.5, minlen=3))

sortedrules_conf <- sort(tweettrans_rules, by="confidence",decreasing=TRUE)
inspect(tweettrans_rules[1:100])


summary(tweettrans_rules)
inspect(head(tweettrans_rules))
inspect(tweettrans_rules[1:5])
#sort the rules
sortedrules_conf <- sort(tweettrans_rules, by="lift",decreasing=TRUE)
inspect(head(sortedrules_conf))


sortedrules_supp <- sort(tweettrans_rules, by="support",decreasing=TRUE)
inspect(head(sortedrules_supp))
inspect(sortedrules_supp[1:500])
#plot the rules
plot(tweettrans_rules[1:100], method="graph", interactive = TRUE, shading="support")
plot(sortedrules_supp[1:100],method="graph",interactive=TRUE,shading="support")



#################################################################################################


#load up
library(arules)
library(arulesViz)
library(FactoMineR)
library(dplyr)
library(e1071)
library(rpart)
library(caret)
library(rpart.plot)
library(rattle)
library(class)
library(randomForest)


######################################################

# association rule mining

######################################################

#choose file
fname <- file.choose()
aus <- read.csv(fname, stringsAsFactors = FALSE)
aus <- na.omit(aus)

#select dimensions
aus1 <- aus[,-c(1, 8 , 11, 13, 15, 17, 19, 21, 22)]

#associative rules
tweettrans_rules = arules::apriori(aus1
                                   ,parameter = list(support=.06, confidence=.5, minlen=3))

sortedrules_conf <- sort(tweettrans_rules, by="confidence",decreasing=TRUE)

inspect(tweettrans_rules[1:5])

plot(tweettrans_rules[1:5], method="graph", interactive = TRUE, shading="support")
plot(sortedrules_conf[1:5],method="graph",interactive=TRUE,shading="support")



######################################################

# sample / cross validation

######################################################


#reduce rows
percent <- .25
set.seed(275)

#create dataframe
Digitsplit <- sample(nrow(aus1), nrow(aus1)*percent)
df <- aus1[Digitsplit,]
str(df)


barplot(table(df$RainTomorrow), main = "freq of labels in dataset")

#RainTomorrow has to be a factor
df$RainTomorrow <- as.factor(df$RainTomorrow)




#crossvalidation
N <- nrow(df)

kfolds <- 3

holdout <- split(sample(1:N), 1:kfolds)



######################################################

# naive bayes

######################################################


#build test and training
Allresults <- list()
Alllabels <- list()
AllAccuracy <- list()


for (k in 1:kfolds) {
  test <- df[holdout[[k]],]
  train <- df[-holdout[[k]],]
  
  #labels
  test_nolabel <- test[-c(14)]
  test_label <- test$RainTomorrow
  
  #model
  model <- naiveBayes(RainTomorrow ~., data=train, na.action = na.pass)
  
  #prediction
  pred <- predict(model, test_nolabel)
  
  #confusion matrix
  cm <- confusionMatrix(pred, test$RainTomorrow)
  #print(cm$table)
  
  acc <- round(cm$overall[1]*100,2)
  AllAccuracy <- c(AllAccuracy, acc)
  
  #accumulate results
  Allresults <- c(Allresults, pred)
  Alllabels <- c(Alllabels, test_label)
  
  #plot
  #plot(pred, ylab = "density", main = "naive bayes plot")
  
}

#confusion matrix and accuracy
table(unlist(Allresults), unlist(Alllabels))
mean(unlist(AllAccuracy))





######################################################

# decision tree

######################################################


#build test and training
Allresults <- list()
Alllabels <- list()
AllAccuracy <- list()


for (k in 1:kfolds) {
  test <- df[holdout[[k]],]
  train <- df[-holdout[[k]],]
  
  #labels
  test_nolabel <- test[-c(14)]
  test_label <- test$RainTomorrow
  
  #model
  model <- rpart(RainTomorrow ~., data=train, method = "class"
                 , control = rpart.control(cp=0.017))
  
  #plots
  fancyRpartPlot(model)
  rsq.rpart(model)
  plotcp(model)
  
  #prediction
  pred <- predict(model, test_nolabel, type = "class")
  
  #confusion matrix
  cm <- confusionMatrix(pred, test$RainTomorrow)
  #print(cm$table)
  
  acc <- round(cm$overall[1]*100,2)
  AllAccuracy <- c(AllAccuracy, acc)
  
  #accumulate results
  Allresults <- c(Allresults, pred)
  Alllabels <- c(Alllabels, test_label)
  
  #plot
  #plot(pred, ylab = "density", main = "decision tree plot")
  
}

#confusion matrix and accuracy
table(unlist(Allresults), unlist(Alllabels))
mean(unlist(AllAccuracy))





######################################################

# svm

######################################################

#choose kernel
kernel_type = "linear"
kernel_type = "polynomial"
kernel_type = "radial"
kernel_type = "sigmoid"


#build test and training
Allresults <- list()
Alllabels <- list()
AllAccuracy <- list()


for (k in 1:kfolds) {
  test <- df[holdout[[k]],]
  train <- df[-holdout[[k]],]
  
  #labels
  test_nolabel <- test[-c(14)]
  test_label <- test$RainTomorrow
  
  #tune the cost
  #tune <- tune(svm, RainTomorrow~., data = train, kernel = kernel_type , ranges = list(cost = c(.01, .1, 1, 10, 100, 1000)))
  
  #tune_best <- round(tune$best.performance, 3)
  
  #model
  model <- svm(RainTomorrow ~., data=train, na.action = na.pass, cost = .5)
  
  #prediction
  pred <- predict(model, test_nolabel, type = c("class"))
  
  #confusion matrix
  cm <- confusionMatrix(pred, test$RainTomorrow)
  #print(cm$table)
  
  acc <- round(cm$overall[1]*100,2)
  AllAccuracy <- c(AllAccuracy, acc)
  
  #accumulate results
  Allresults <- c(Allresults, pred)
  Alllabels <- c(Alllabels, test_label)
  
  #plot
  #plot(pred, ylab = "density", main = "naive bayes plot")
  
}

#confusion matrix and accuracy
table(unlist(Allresults), unlist(Alllabels))
mean(unlist(AllAccuracy))





######################################################

# kNN

######################################################

#kNN can only use quantitative data
df1 <- df
df1$Location <- NULL
df1$WindDir9am <- NULL
str(df1)


#choose k
k_guess = 3


#build test and training
Allresults <- list()
Alllabels <- list()
AllAccuracy <- list()


for (k in 1:kfolds) {
  test <- df1[holdout[[k]],]
  train <- df1[-holdout[[k]],]
  
  #labels
  test_nolabel <- test[-c(12)]
  test_label <- test$RainTomorrow
  
  train_nolabel <- train[-c(12)]
  train_label <- train$RainTomorrow
  
  #model
  model <- knn(train = train_nolabel, test = test_nolabel, cl = train_label
               , k = k_guess, prob = FALSE)
  
  #confusion matrix
  cm <- confusionMatrix(model, test$RainTomorrow)
  #print(cm$table)
  
  acc <- round(cm$overall[1]*100,2)
  AllAccuracy <- c(AllAccuracy, acc)
  
  #accumulate results
  Allresults <- c(Allresults, model)
  Alllabels <- c(Alllabels, test_label)
  
  #plot
  #plot(pred, ylab = "density", main = "naive bayes plot")
  
}

#confusion matrix and accuracy
table(unlist(Allresults), unlist(Alllabels))
mean(unlist(AllAccuracy))





######################################################

# random forest

######################################################

#Number of variables randomly sampled as candidates at each split
mtry_value = 8

#build test and training
Allresults <- list()
Alllabels <- list()
AllAccuracy <- list()


for (k in 1:kfolds) {
  test <- df[holdout[[k]],]
  train <- df[-holdout[[k]],]
  
  #labels
  test_nolabel <- test[-c(14)]
  test_label <- test$RainTomorrow
  
  #model
  model <- randomForest(RainTomorrow ~., data=train, importance = T, mtry = mtry_value)
  
  #prediction
  pred <- predict(model, test_nolabel)
  
  #confusion matrix
  cm <- confusionMatrix(pred, test$RainTomorrow)
  #print(cm$table)
  
  acc <- round(cm$overall[1]*100,2)
  AllAccuracy <- c(AllAccuracy, acc)
  
  #accumulate results
  Allresults <- c(Allresults, pred)
  Alllabels <- c(Alllabels, test_label)
  
  #plot
  #plot(pred, ylab = "density", main = "naive bayes plot")
  
}

#confusion matrix and accuracy
table(unlist(Allresults), unlist(Alllabels))
mean(unlist(AllAccuracy))