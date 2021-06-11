library(ggplot2)
library(tidyverse)
library(dplyr)
library(sqldf)
library(RColorBrewer)

info = file.choose()
chess = read.csv(info, stringsAsFactors = FALSE)

#check for missing data
any(is.na(chess))

#remove games that were not rated
table(chess$rated)
chess$rated = gsub("False", "FALSE", chess$rated)
chess$rated = gsub("True", "TRUE", chess$rated)
chess <- chess[chess$rated != "FALSE",]

#check structure of dataset
str(chess)

#create ratings for each game based on each player's rating
chess$avg_rating <- (chess$white_rating + chess$black_rating)/2
range(chess$rating_average)

#bargraph of most frequent openings for highly rated players
all_matches <- sqldf('SELECT turns, winner,  victory_status, white_rating, black_rating, avg_rating, opening_name
           , COUNT(opening_name) AS count FROM chess WHERE avg_rating > 1900 GROUP BY opening_name ORDER BY COUNT(opening_name)')
nrow(all_matches)


sample_matches <- sqldf('SELECT turns, winner,  victory_status, white_rating, black_rating, avg_rating, opening_name
           , COUNT(opening_name) AS count FROM chess GROUP BY opening_name ORDER BY COUNT(opening_name) DESC LIMIT 10')
nrow(sample_matches)


ggplot(sample_matches) + aes(reorder(x=opening_name, count), count)+geom_col(fill = "#f3dbb4")+coord_flip()+
  theme_minimal()+theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 10))+
  labs(y = "count", x = "", color = "precipitation\n(inches)",
       title = "Frequent Openings")

#How were the games spread between Victory status and match winners?
# Visualizing Victory Status and Match Winners
ggplot(chess) + aes(victory_status, winner, color = turns)+geom_jitter(alpha=.4)+ 
  scale_color_gradient(low = "#f3dbb4", high = "#856048")+theme_minimal()+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 10))+
  labs(y = "Winner", x = "Victory Status", color = "turns",
       title = "Victory Status and Match Winners")


#piechart of game ratings
#colfunc <- colorRampPalette(c("#296d98", "#ADD8E6"))
colfunc <- colorRampPalette(c( "#c93f6d","#f3dbb4"))
chess$game_rank <- ifelse(chess$avg_rating > 1900, "High\n(14%)", "Moderate\n(62%)")
chess$game_rank[chess$avg_rating < 1400] <- "Low\n(24%)\n"
pie(table(chess$game_rank), col = colfunc(3))

#percentages
nrow(chess[chess$game_rank == "Low rating",])/nrow(chess)
nrow(chess[chess$game_rank == "Moderate rating",])/nrow(chess)
nrow(chess[chess$game_rank == "High rating",])/nrow(chess)


#rating plot with difference between ratings
chess$diff <- abs(chess$white_rating - chess$black_rating)
ggplot(chess) + aes(white_rating, black_rating, color = diff)+geom_point(alpha = .9)+
  scale_color_gradient(low = "#f3dbb4", high = "#856048")+theme_minimal()+
  geom_smooth(method = lm, color = "#296d98", size = 1)

#boxplot
chess$game_rank <- ifelse(chess$avg_rating > 1900, "High rating", "Moderate rating")
chess$game_rank[chess$avg_rating < 1400] <- "Low rating"


ggplot(chess) + aes(reorder(x= game_rank, turns), turns)+geom_boxplot(fill = colfunc(3))+
  theme_minimal()+theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 10))+
  labs(y = "turns", x = "", color = "",
       title = "Distribution of Victory Status by Ratings")

#histogram
colfunc <- colorRampPalette(c("#f3dbb4", "#c93f6d"))
ggplot(chess) + aes(x= avg_rating)+geom_histogram(bins = 10, fill = colfunc((10)))+
  theme_minimal()+theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 10))+
  labs(y = "count", x = "Match Rating", color = "",
       title = "")





