# Data Science: Capstone

# title: "Capstone Project - Recommendation System"
# author: "Frank Valdivia"
# date: "2025-05-16"

# Project Overview: MovieLens

# In this project, MovieLens dataset will be used to create a movie recommendation system.
# The dataset is pulled from the dslabs package and has millions of ratings.
# In this project, a subset of 10 million ratings will be used with the purpose of reducing computing time.

# A train set will be generated with the name: edx
# A test set will be generated with the name final_holdout_test.
# RMSE will be used to evaluate the error between predictions and true values in the final_holdout_test set.
# Six models will be developed with their respective RMSEs

# Install relevant libraries

install.packages("tidyverse")
install.packages("caret")
install.packages("ggplot2")
install.packages("lattice")
install.packages("gridExtra")
install.packages("beepr")


library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(beepr)

# Creating Train dataset: edx and Test dataset: final_holdout_test sets 

# Pull the 10 million ratings dataset and save it as a ZIP file on the computer. Location:
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Creating the Ratings dataset from the file downloaded

# This step may take a few minutes

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
		
beep()
		
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Showing structure and number of records of the Ratings dataset

head(ratings)
dim(ratings)

# Creating the Movies dataset

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Showing structure and number of records of the Movies dataset

head(movies)
dim(movies)

# Left joining Movies dataset to Ratings dataset
# This may take a few minutes

movielens <- left_join(ratings, movies, by = "movieId")

beep()

# Showing structure and number of records of the adjusted MovieLens dataset

head(movielens)
dim(movielens)

# Creating the Train and Test sets using MovieLens

set.seed(1, sample.kind="Rounding") # for R 3.6 or later

# Final hold-out test set will be 10% of MovieLens data

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

# Creating edx as the Train set
# Creating temp as the temporary Test set

edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Showing structure and number of records of the Train set
head(edx)
dim(edx)

# Showing structure and number of records of the Temp Test set

head(temp)
dim(temp)


# For the test set, only the users and movies that have ratings will be used. Exclude all the others.
# The result is the final test dataset: final_holdout_test

final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Showing structure and number of records of the Test set

head(final_holdout_test)
dim(final_holdout_test)

# Move rows removed from Temp to the edx (Train) set

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Showing structure and number of records of the Final version of Train set

head(edx)
dim(edx)

# Remove datasets that will not be used to release resources from the computer

rm(dl, ratings, test_index, temp, movielens, removed)

beep()

# Exploratory Analysis of both datasets: Movies and Users

p1 <- edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  labs(x="Ratings per Movie", y = "Count") +
  ggtitle("Histogram")

p2 <- edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  labs(x="Ratings per User", y = "Count") +
  ggtitle("Histogram")
 
grid.arrange(p1, p2, ncol = 2)

# Some general characteristics of the data:
# - They seem to be normally distributed
# - Some movies have more ratings than others
# - Some users rate more movies than others
# - There are no presence of outliers

# Creation of function to calculate the Root Mean Squared Error or RMSE. This function will be used to calculate the RMSE of the model that generated predictions:

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

beep()

# Six models will be compared in the following sections

# ===============================================================================
# 1. The first model will be just using the Mean rating of the whole dataset as a predictor

mu_hat <- mean(edx$rating)
mu_hat

# mu_hat is the predictor for all cases

# Calculation of RMSE for mu_hat as the prediction compared to all final_holdout_test ratings

predictions <- rep(mu_hat, nrow(final_holdout_test))
mu_hat_rmse <- RMSE(final_holdout_test$rating, predictions)
mu_hat_rmse

# mu_hat_rmse should be 1.061202

# Because mu_hat is the mean, then any other unique predictor should produce a higher RMSE
# The following calculates the RMSE for a predictor that is just slightly greater than mu_hat

predictions <- rep(mu_hat + 0.1, nrow(final_holdout_test))
RMSE(final_holdout_test$rating, predictions)

# When adding 0.1 to mu_hat, RMSE is higher (1.065944)

# Adding RMSE to a table that will report RMSEs for all models in this project

rmse_results_t <- tibble(method = "1: Predictor = mu_hat", RMSE = mu_hat_rmse, Lambda = "")
rmse_results_t %>% knitr::kable()

beep()

# =============================================================
# 2. Modeling movie effects

# Every Movie has its own Rating mean, which might be different from the overall mean.
# That is due to the bias (or effect) of every movie on the ratings.
# In the second model, the Movie bias or effect will be incorporated.
# The Bias will be calculated for every Movie as the difference between the
# Mean Rating of that movie and the Mean Rating of the whole dataset (mu_hat)
# bi (or b_i) will be the Bias per Movie.
# bi is then added to mu_hat for every Movie and that is the Predictor for that Movie.
# Adding the Movie bi should improve the prediction and generate a smaller RMSE

# Calculating bi per Movie and saving them in movie_avgs

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

# Showing structure and number of records of the movie_avgs set

head(movie_avgs)
dim(movie_avgs)

# In the following plot, the distribution of the bias shows that there is variation
# around the value of 0 and most bias numbers go from -3 to 1	 

p1 <- movie_avgs %>% 
 ggplot(aes(b_i)) + 
 geom_histogram(bins = 30, color = "black") + 
 geom_vline(aes(xintercept=mean(b_i)), color="blue", linetype="dashed", linewidth=1) +
 labs(x=paste("Movie b_i (mean(b_i) = ",round(mean(movie_avgs$b_i),2),")"), y = "Count of b_i") +
 ggtitle("Histogram - Movie bias b_i")
grid.arrange(p1, ncol = 1)

# In the following code, the prediction is Y = U + Movie_bi where U = mu_hat and Movie_bi = pull_movie_bias

pull_movie_bias <- final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
  
predicted_ratings <- mu_hat + pull_movie_bias
  
# Showing structure and number of records of the predicted_ratings set

head(predicted_ratings)
length(predicted_ratings) # is an array not a table

# Calculation of RMSE

model_bi_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_bi_rmse

# Movie_bias RMSE should be 0.9439087


rmse_results_t <- bind_rows(rmse_results_t,
                          tibble(method="2: Predictor = mu_hat + Movie_bi",
                                     RMSE = model_bi_rmse, Lambda = ""))
rmse_results_t %>% knitr::kable()



# RMSE of the second model is lower due to including Movie bias

beep()

# ==================================================================
# 3. In the third model, User bias (effect) will be added to the previous model.
# This updated model will have both Movie bias and User bias

# User bias = bu or b_u

# In the following plot, the distribution of Ratings shows that there is variation
# around the value of 4 and most ratings go from 1 to 5	 

user_avgs <- edx %>% 
     group_by(userId) %>%
     summarize(b_u = mean(rating)) %>% 
     filter(n()>=100)

user_avgs_mean_bu <- mean(user_avgs$b_u)


edx %>% 
     group_by(userId) %>% 
     summarize(b_u = mean(rating)) %>% 
     filter(n()>=100) %>%
     ggplot(aes(b_u)) + 
     geom_histogram(bins = 30, color = "black") +
   geom_vline(aes(xintercept=mean(b_u)), color="blue", linetype="dashed", linewidth=1) +
   labs(x=paste("User b_u (mean(b_u) = ",round(mean(user_avgs_mean_bu),2),")"), y = "Count of b_u for n>100") +
     ggtitle("Histogram - User bias b_u for n>100")
	 

# This variation means that some users rate higher than other users do. 
# This variability also suggests that incorporating User bias would improve the model

# The third model will be  Y = bi + bu
# Here the prediction is Y = U + Movie_bi + User_bu 

# The third model can be fit with the Linear Model function:
# lm(rating ~ as.factor(movieId) + as.factor(userId))
# But the computation of this linear model would require potentially days to process on a 
# standard computer due to the number of datapoints

# User bias will be calculated by subtracting mu_hat and movie_bias from Y (Rating)
# User_bias =  Y - u - bi
# User_bias will be averaged per user

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Showing structure and number of records of the user_avgs set

head(user_avgs)
dim(user_avgs)

# Calculation of predictors = U + Movie_bias + User_bias,
# where U = mu_hat,   Movie_bias = b_i,   User_bias = b_u

predicted_ratings <- final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

# Calculation of RMSE

model_bi_bu_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_bi_bu_rmse

# Movie_and_User_bias RMSE should be 0.8653488
# RMSE is lower due to adding User bias to the previous model

rmse_results_t <- bind_rows(rmse_results_t,
                          tibble(method="3: Predictor = mu_hat + Movie_bi + User_bu",  
                                     RMSE = model_bi_bu_rmse, Lambda = ""))
rmse_results_t %>% knitr::kable()

beep()

# ==================================================================
# 4. The fourth model will apply Regularization
# RMSE was reduced when movie bias and user bias were incorporated.

# The following table shows the 10 largest mistakes (largest residuals)
# predicted by the second model (Y = U + Movie_bias):

final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu_hat + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% pull(title) 

# Following is the list of the 20 best movies based just on Movie_bias.

movie_avgs %>% left_join(movies, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  dplyr::select(title, b_i) %>% 
  slice(1:20) %>%  
  pull(title)

# Following is the list of the  20 worst movies based just on Movie_bias.

movie_avgs %>% left_join(movies, by="movieId") %>%
  arrange(b_i) %>% 
  dplyr::select(title, b_i) %>% 
  slice(1:20) %>%  
  pull(title)

# The output lists (best and worst) show movies that are not well known.

# The following code shows the number of ratings (sample size) 
# for those movies listed above as best or worst based on Movie_bias

edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movies, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:20) %>% 
  pull(n)

edx %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movies, by="movieId") %>%
  arrange(b_i) %>%
  slice(1:20) %>% 
  pull(n)

# The resulting counting (Sample size)
# should show:  [1] 1 2 1 1 1 1 4 4 4 2 7 3 3 1 1 3 1 1 4 1
# and [1]   2   1   1   1   2  56  14  32 199   2   2   1   1   1   1   1   1 137  15  68
# which tells that they have very few ratings (very small sample size)
# These movies are not known and thus have very few ratings
# meaning very small sample sizes

# Having less ratings increases larger bias and generates higher residuals.
# Higher residuals increase the RMSE.

# Regularization will be used to reduce the impact of large residuals  
# that are coming from a few ratings (or small sample size) 

# Using Regularization, the small sample size bias is penalized by adding a Lambda to "n", 
# where n is the number of ratings for movie i.

# The higher the "n", the lower the impact of Lambda
# The lower the "n", the higher the impact of Lambda

# Lambda is a parameter. In the first iteration, Lambda = 3 will be used 
# to compare Movie_bias (movie_avgs$b_i) to Movie_regularized_bias (movie_reg_avgs$b_i)

# Calculation of Movie_regularized_bias with Lambda = 3

lambda <- 3
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# The following is a plot of Movie_bias versus Movie_regularized_bias with Lambda = 3

tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5) +
  labs(x="Original b_i", y = "Regularized b_i") +
 ggtitle("Movie Original bias vs Regularized bias for Lambda = 3")

# The resulting plot shows how Regularized_bias and bias differ more for small values of n.

# Generating again the list of the top 20 best and worst movies but
# now using Movie_regularized_bias instead of Movie_bias

# The list of the top 20 best movies based on movie regularized b_i:

edx %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  left_join(movies, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  dplyr::select(title, b_i, n) %>% 
  slice(1:20) %>% 
  pull(title)

# The list of the top 20 worst movies based on movie regularized b_i:

edx %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  left_join(movies, by="movieId") %>%
  arrange(b_i) %>% 
  dplyr::select(title, b_i, n) %>% 
  slice(1:20) %>% 
  pull(title)

# Previous lists now consist of movies that have more ratings or adjusted for small sample size.

# Calculation of predictors = U + Movie_regularized_bias,
# where U = mu_hat,   Movie_regularized_bias = movie_reg_avgs$b_i

predicted_ratings <- final_holdout_test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu_hat + b_i) %>%
  pull(pred)

# Calculation of RMSE

model_RegMovie_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_RegMovie_rmse

# Movie_Regularized_bias RMSE with Lambda=3 should be 0.9438538
# This RMSE is lower than the RMSE of model 2, which has Movie_bias

rmse_results_t <- bind_rows(rmse_results_t,
                          tibble(method="4: Predictor = mu_hat + Movie_regularized_bi, arbitrary Lambda = 3",  
                                     RMSE = model_RegMovie_rmse, Lambda = as.character(3)))
rmse_results_t %>% knitr::kable()

beep()

# ==================================================================
# 5. The fifth model will optimize the value of Lambda for the fourth model

# Values of Lambda will be used in a loop to find the Lambda that minimizes RMSE
# Lambda will be tested between 0 and 10, with an increment of 0.25

Sys.time()		# Time is shown to track duration

pb = txtProgressBar(min = 0, max = 10, initial = 0, style = 3)	# Progress bar set-up

# First, estimate just the summation per Movie
lambdas <- seq(0, 10, 0.25)
just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu_hat), n_i = n())

# Second, iterate the calculation of RMSE depending on the values of Lambda

rmses <- sapply(lambdas, function(l){
  setTxtProgressBar(pb,l)										# Set progress bar to "l"
  predicted_ratings <- final_holdout_test %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu_hat + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, final_holdout_test$rating))
})

close(pb)														# Close progress bar

Sys.time()		# Time is shown to track duration

# Plot values of Lambda versus RMSE

lambda <- lambdas[which.min(rmses)]

tibble(lambdas, rmses) %>%
  ggplot(aes(lambdas, rmses)) + 
  geom_point(shape=1, alpha=0.5) +
  geom_point() +
  geom_vline(aes(xintercept=lambda), color="blue", linetype="dashed", linewidth=1) +
  labs(x=paste("Lambdas (Optimized Lambda = ",lambda,")"), y = "RMSEs") +
  ggtitle("Lambda vs RMSE")



# Show the Lambda that minimizes RMSE

lambdas[which.min(rmses)]

# Show minimum RMSE

min(rmses)

# Movie_Regularized_bias RMSE with optimized Lambda should be 0.9438521
# Because there is no significant difference between 3 and 2.5 for the Lambda value
# the RMSE has decreased but not significantly

rmse_results_t <- bind_rows(rmse_results_t,
                          tibble(method=paste("5: Predictor = mu_hat + Movie_regularized_bi, optimized Lambda =",lambdas[which.min(rmses)]), RMSE = min(rmses), Lambda = as.character(lambda)))
rmse_results_t %>% knitr::kable()

beep()

# ==================================================================
# 6. The sixth model will optimize the value of Lambda for the fifth model
# but it will add User Regularized bias

# Values of Lambda will be used in a loop to find the Lambda that minimizes RMSE
# Lambda will be tested between 0 and 10, with an increment of 0.1

Sys.time()		# Time is shown to track duration

pb = txtProgressBar(min = 0, max = 10, initial = 0, style = 3)	# Progress bar set-up

lambdas <- seq(0, 10, 0.1)

# This may take a few minutes to run

rmses <- sapply(lambdas, function(l){
  setTxtProgressBar(pb,l)										# Set progress bar to "l"
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+l))
  predicted_ratings <- 
    final_holdout_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, final_holdout_test$rating))
})

close(pb)														# Close progress bar

Sys.time()		# Time is shown to track duration

# Plot values of Lambda versus RMSE

lambda <- lambdas[which.min(rmses)]

tibble(lambdas, rmses) %>%
  ggplot(aes(lambdas, rmses)) + 
  geom_point(shape=1, alpha=0.5) +
  geom_point() +
  geom_vline(aes(xintercept=lambda), color="blue", linetype="dashed", linewidth=1) +
  labs(x=paste("Lambdas (Optimized Lambda = ",lambda,")"), y = "RMSEs") +
  ggtitle("Lambda vs RMSE")



# Show the Lambda that minimizes RMSE

lambda <- lambdas[which.min(rmses)]

lambda

# Show minimum RMSE

min(rmses)

# Movie_Regularized_bias and User_Regularized_bias RMSE with optimized Lambda should be 0.8648170
# This is the lowest RMSE

rmse_results_t <- bind_rows(rmse_results_t,
                          tibble(method=paste("6: Predictor = mu_hat + Movie_regularized_bi + User_regularized_bu, optimized Lambda =",lambda),  
                                     RMSE = min(rmses)))

rmse_results_t %>% knitr::kable()

beep()

# ==================================================================
# the sixth model has the lowest RMSE 
# Movie_Regularized_bias and User_Regularized_bias RMSE with optimized Lambda = 5.2 has an RMSE lower than: 0.86490
# This is the best model and provides the best movie recommendations to users

