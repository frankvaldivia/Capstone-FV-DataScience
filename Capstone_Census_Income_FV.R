# title: "HarvardX Capstone My Own Project - US Census Data"
# author: "Frank Valdivia"
Sys.Date()

# INTRODUCTION
 
# The goal of this project is to build a CLASSIFICATION model that will predict 
# INCOME LEVEL for the 1994 US Census Data (1) downloaded from The UCI Machine Learning Repository. 
# Six classification models will be built and each model assessed against its Accuracy.
# The goal is to achieve more than 80% Accuracy and the model with the hightest Accuracy on the train set will be selected. 

# The US Census dataset has 14 columns of which "Income Level" is the outcome. Income has two levels: "<=50K" ">50K" 

# According to the UCI website (https://archive.ics.uci.edu/dataset/2/adult):

# "Extraction was done by Barry Becker from the 1994 US Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))"

# The following is the list of the other features that the dataset has:

# - age: continuous.

# - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.

# - fnlwgt: continuous.

# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

# - education-num: continuous.

# - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

# - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

# - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

# - sex: Female, Male.

# - capital-gain: continuous.

# - capital-loss: continuous.

# - hours-per-week: continuous.

# - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# A subset of features will be used to build the model

########## IMPORTANT ############
#
# To run this report, the file adult.zip from the UCI website should be in your working directory.
# I copied the adult.zip file from the UCI website (UC Irvine Machine Learning Repository) to mu GutHub folder.
# This file will be downloaded to your working folder.

options(timeout = 120)

adult_zip <- "adult.zip"
if(!file.exists(adult_zip))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/adult.zip", destfile = adult_zip)

# It may happen that the download does not work.
# if that is the case, please go to my GitHub and download adult.zip manually and copy it to your working directory
# Here is the github address:
# 
# https://github.com/frankvaldivia/Capstone-FV-DataScience/tree/main
#
# Or
# 
# Download it from the UCI website
# https://archive.ics.uci.edu/dataset/2/adult
#
#
####### END OF IMPORTANT ########


########## IMPORTANT ############
# Because processing time to generate the models was significant (between hours and more than one day each), 
# the  six models evaluated were generated and saved as RDS files in GitHub. 
# Generating the models takes hours or more than a day.
# Downloading the models saves significant time.

# Download the six models from GitHub into your working directory:\

# train_knn.rds: KNN model \
# train_knn_cv.rds: KNN model with Cross Validation Model \
# train_dt.rds: Decision Tree model \
# train_dt_cp.rds: Decision Tree - Complexity Parameter model \
# train_rf.rds: Random Forest model\
# train_rf_2.rds: Random Forest wth Cross Validation model\

options(timeout = 120)

model <- "train_knn.rds"
if(!file.exists(model))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/train_knn.rds", destfile = model)

options(timeout = 120)

model <- "train_knn_cv.rds"
if(!file.exists(model))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/train_knn_cv.rds", destfile = model)

options(timeout = 120)

model <- "train_dt.rds"
if(!file.exists(model))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/train_dt.rds", destfile = model)

options(timeout = 120)

model <- "train_dt_cp.rds"
if(!file.exists(model))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/train_dt_cp.rds", destfile = model)

options(timeout = 120)

model <- "train_rf.rds"
if(!file.exists(model))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/train_rf.rds", destfile = model)

options(timeout = 120)

model <- "train_rf_2.rds"
if(!file.exists(model))
  download.file(url = "https://github.com/frankvaldivia/Capstone-FV-DataScience/raw/refs/heads/main/train_rf_2.rds", destfile = model)

 
# You can download the RDS files manually from my GitHub
# 
# https://github.com/frankvaldivia/Capstone-FV-DataScience/tree/main
#
####### END OF IMPORTANT ########


########## IMPORTANT ############
 
# Following, the user gets a prompt to answer if he/she wants to use the downloaded models or to generate the models when runnng the R code.
# IMPORTANT: It is recommended to select YES to use the downloaded models because generating takes minutes to a day for each model

answerYes <- askYesNo("Yes: to run using the models already generated and downloaded;      No: to generate models - may take hours")

if (is.na(answerYes)) {
  # Code to execute if the user click on Cancel or Esc, The user has to go back and answer the question
  print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
  # Code to execute if the user answers "yes"
  print("Running from generated models downloaded from GitHub...")
} else if (answerYes == FALSE) {
  # Code to execute if the user answers "no"
  print("Generating models...This may take hours")
} else {
  # Code to handle invalid input. this should not happen becuase NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

####### END OF IMPORTANT ########

 
# Machine learning methods will be used, which means that two sub datasets will be created during this process:

# A train set will be generated with the name: censusTrain; this dataset will be used to train the model.

# A test set will be generated with the name:  censusTest; this dataset will be used to test the model trained using the train dataset.

# The output will be a classification model and Accuracy will be used to measure quality of the model.

# ANALYSIS AND METHODS

# The overall process is as follows:

# 2.1. Loading libraries

# 2.2. Unzipping donloaded file and Generating datasets

# 2.3. Analysis and methods


## Loading libraries

# To run the report the following R libraries need to be previously installed:

# - library(tidyverse)
# - library(caret)
# - library(ggplot2)
# - library(dplyr)
# - library(lattice)
# - library(gridExtra)
# - library(beepr)
# - library(dslabs)
# - library(rpart)
# - library(rpart.plot)
# - library(randomForest)
# - library(Rborist)
# - library(class)
# - library(grid)

# Above libraries are loaded.

if (!require("tidyverse")) install.packages("tidyverse")
if (!require("caret")) install.packages("caret")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("lattice")) install.packages("lattice")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("beepr")) install.packages("beepr")
if (!require("dslabs")) install.packages("dslabs")
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("randomForest")) install.packages("randomForest")
if (!require("Rborist")) install.packages("Rborist")
if (!require("class")) install.packages("class")

library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(lattice)
library(gridExtra)
library(beepr)
library(dslabs)
library(rpart)
library(rpart.plot)
library(randomForest)
library(Rborist)
library(class)
library(grid)

# +++++++ IMPORTANT +++++++

# Because processing time to generate the models was significant (more than one day), the  six models evaluated were generated and saved as RDS files in GitHub and downloaded into your working directory:\

# train_knn.rds: KNN model \
# train_knn_cv.rds: KNN model with Cross Validation Model \
# train_dt.rds: Decision Tree model \
# train_dt_cp.rds: Decision Tree - Complexity Parameter model \
# train_rf.rds: Random Forest model\
# train_rf_2.rds: Random Forest wth Cross Validation model\

# GitHub link:\

# https://github.com/frankvaldivia/Capstone-FV-DataScience/tree/main

# +++++++++++++++++++++++++

beep()

# adult.zip has two files, The author already generated a train dataset and a test dataset. In this project, both files will be put together to create a whole Census_data dataset, which will be used to create a new train dataset and a new test dataset.


dl <- "adult.zip"

options(timeout = 120)

#  From adult.zip, extract the data sets
#  loading original train data file

adult_data_file <- "adult.data"
if(!file.exists(adult_data_file))
  unzip(dl, adult_data_file)

#  loading original test data file

adult_test_file <- "adult.test"
if(!file.exists(adult_test_file))
  unzip(dl, adult_test_file)

# both files will be put together to generate new train and test sets

# Loading into data frames and cleaning

# This step may take a few minutes

census_train <- as.data.frame(str_split(read_lines(adult_data_file), fixed(","), simplify = TRUE), stringsAsFactors = TRUE)

census_test <- as.data.frame(str_split(read_lines(adult_test_file), fixed(","), simplify = TRUE), stringsAsFactors = TRUE)

# Following, 

# 1. Column names are assigned.
# 2. Empty rows are removed.
# 3. Temporary train and test data sets are created.

beep()

# assigning column names to both temporary files

colnames(census_train) <- c("age", "workclass", "weight", "education", 
                            "educationlevel", "maritalstatus", "occupation", 
                            "relationship", "race", "sex", "capitalgain", 
                            "capitalloss", "hoursperweek", "nativecountry", 
                            "income")

colnames(census_test) <- c("age", "workclass", "weight", "education",
                           "educationlevel", "maritalstatus", "occupation",
                           "relationship", "race", "sex", "capitalgain",
                           "capitalloss", "hoursperweek", "nativecountry",
                           "income")

# removing empty rows

temp_train<-census_train[-c(32562,32563),]

temp_test<-census_test[-c(1,16283,16284),]

# Temporary train and test data sets are created with the following structure and number of rows. These two data sets were originally created by the author and will be used to create a new master census data for this project.

# Showing structure and number of records of the temp_train dataset

cat("\nTemporary train set: Columns\n")

colnames(temp_train) %>% knitr::kable()

cat("\nTemporary train set: Number of Records\n")

tibble(dim(temp_train))[1,1] %>% knitr::kable()

# Showing structure and number of records of the temp_test dataset

cat("\nTemporary test set: Columns\n")

colnames(temp_test) %>% knitr::kable()

cat("\nTemporary test set: Number of Records\n")

tibble(dim(temp_test))[1,1] %>% knitr::kable()

# generating the new master dataset

census_data <- rbind(temp_train,temp_test)

# Showing structure and number of records of the Census dataset

cat("\nNew Census data: Columns\n")

colnames(census_data) %>% knitr::kable()

cat("\nNew Census data: Number of Records\n")

tibble(dim(census_data))[1,1] %>% knitr::kable()

# With the whole new Census Data set, predictors (columns) are redefined as Factors or Integers. The outcome (Y) was redefined as a string with two values:  “<=50K” “>50K”

# Cleaning Census data continued with the removal of columns that will not be used in the analysis, empty records and the values for the outcome.

# The census data set to be used in this project is as follows:

# converting to factors

census_data <- census_data %>%
  mutate(age = as.integer(age),
  	  workclass = factor(str_trim(as.character(workclass))),
      weight = as.integer(weight),
		  education = factor(str_trim(as.character(education))),
		  educationlevel = as.integer(educationlevel),
		  maritalstatus = factor(str_trim(as.character(maritalstatus))),
		  occupation = factor(str_trim(as.character(occupation))),
		  relationship = factor(str_trim(as.character(relationship))),
		  race = factor(str_trim(as.character(race))),
		  sex = factor(str_trim(as.character(sex))),
		  capitalgain = as.integer(capitalgain),
		  capitalloss = as.integer(capitalloss),
		  hoursperweek = as.integer(hoursperweek),
		  nativecountry = factor(str_trim(as.character(nativecountry))),
		  income = str_trim(as.character(income))
		  )
		
# cleaning up outcome variable: income
		
census_data <- census_data %>%
  mutate(income = gsub("<=50K.","<=50K",income))

census_data <- census_data %>%
  mutate(income = gsub(">50K.",">50K",income))

census_data <- census_data %>%
  mutate(income = factor(income))
  
# Removing variables that will not be included in the analysis

census_data = subset(census_data, select = -c(weight, educationlevel, capitalgain,capitalloss,nativecountry))

# Reordering categories for Education and Relationship

census_data$education <- factor(census_data$education, levels = c("Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college", "Bachelors", "Masters","Prof-school",  "Doctorate"))

census_data$relationship <- factor(census_data$relationship, levels = c("Husband", "Wife", "Unmarried", "Not-in-family", "Other-relative", "Own-child"))

# Showing structure and number of records of the census_data dataset

cat("\nClean Census data: Columns\n")

colnames(census_data) %>% knitr::kable()

cat("\nclean Census data: Head\n")

head(census_data) %>% knitr::kable()

cat("\nclean Census data: Number of Records\n")

tibble(dim(census_data))[1,1] %>% knitr::kable()

cat("\nclean Census data: levels for every predictor and outcome(Income)\n")

census_data %>% sapply(levels)

## Analysis and methods

# First, exploratory analysis will be carried out showing frequencies for predictors and the outcome (Income).

# Then, a description of the measure of success (accuracy) will be included before elaborating on the models applied to Census data.

# Every model will include specific results and plots and all will have their Accuracy estimated and compared.

# The goal is to have a model with 80% accuracy or greater.

### Exploratory Analysis

# Following are the frequencies for the following predictors. They are all categories:

# - Work class
# - Education
# - Marital Status
# - Race

beep()

# Exploratory analysis

frequency_table <- table(census_data$workclass, census_data$income)

# Generation of frequencies for predictors

p1 <- as.data.frame(frequency_table)
colnames(p1) <- c("Category", "Income", "Frequency")

g1 <- ggplot(p1, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none")  +
	ggtitle("Work class")

frequency_table <- table(census_data$education, census_data$income)

p2 <- as.data.frame(frequency_table)
colnames(p2) <- c("Category", "Income", "Frequency")

g2 <- ggplot(p2, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none")  +
	ggtitle("Education")

frequency_table <- table(census_data$maritalstatus, census_data$income)

p3 <- as.data.frame(frequency_table)
colnames(p3) <- c("Category", "Income", "Frequency")

g3 <- ggplot(p3, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none")  +
	ggtitle("Marital Status")

frequency_table <- table(census_data$race, census_data$income)

p4 <- as.data.frame(frequency_table)
colnames(p4) <- c("Category", "Income", "Frequency")

g4 <- ggplot(p4, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))  +
	ggtitle("Race")

grid.arrange(g1, g2, g3, g4, ncol = 2, 
               top="Frequencies", bottom="Census Data", 
               left=" ", right=" ")	
grid.rect(gp=gpar(fill=NA))

# Previous plots suggest that there is some level of dependancy of income on some specific categories, i.e. HS-Grad, Bachelors, Some College in Education, or Private in Work class.

# Following are the frequencies for the following predictors. They are all categories:

# - Relationship
# - Sex
# - Occupation
# - Income

frequency_table <- table(census_data$relationship, census_data$income)

p5 <- as.data.frame(frequency_table)
colnames(p5) <- c("Category", "Income", "Frequency")

g5 <- ggplot(p5, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none")  +
	ggtitle("Relationship")

frequency_table <- table(census_data$sex, census_data$income)

p6 <- as.data.frame(frequency_table)
colnames(p6) <- c("Category", "Income", "Frequency")

g6 <- ggplot(p6, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))  +
	ggtitle("Sex")

frequency_table <- table(census_data$occupation, census_data$income)

p7 <- as.data.frame(frequency_table)
colnames(p7) <- c("Category", "Income", "Frequency")

g7 <- ggplot(p7, aes(x=Category, y=Frequency, fill=Income)) + 
    geom_bar(stat = "identity") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none")  +
	ggtitle("Occupation")
	
frequency_table <- table(census_data$income)

p8 <- as.data.frame(frequency_table)
colnames(p8) <- c("Category", "Frequency")

g8 <- ggplot(p8, aes(x=Category, y=Frequency)) + 
    geom_bar(stat = "identity", fill ="darkgoldenrod3") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))  +
	ggtitle("Income")

grid.arrange(g5, g6, g7, g8, ncol = 2, 
               top="Frequencies", bottom="Census Data", 
               left=" ", right=" ")	
grid.rect(gp=gpar(fill=NA))

# Previous plots suggest that there is some level of dependancy of income on some specific categories, i.e. Female in Sex, or Own-child in Relationshp.

# The following graph shows the values of Income for the whole Census dataset. Because this is a two dimensional graph, only two predictors out of the nine are shown, Education and Relationship.

# Every dot shows the frequency of occurrences that have ">50K" as income in blue as a percentage of the total number of records for that combination of Education and Relationship. The size of the dot represents the number of records for that combination of Education and Relationship.

#  x = "Education", y = "Relationship"

df_income <- data.frame(census_data[, 3], census_data[, 6], census_data[,10])
colnames(df_income) <- c("Education", "Relationship", "Income")

# print(df_income)

proportion_df <- df_income %>%
  group_by(Relationship, Education) %>%
  summarise(
    count_filtered = sum(Income == ">50K"),
    total_count = n(),
    Red_to_Blue = count_filtered / total_count,
    .groups = "drop"
  )

# print(proportion_df)

base_income_plot <- ggplot(proportion_df, aes(x = Education, y = Relationship, 
                          color = Red_to_Blue, 
                          size = total_count)) +
    geom_point() +
    scale_color_gradient(low = "red", high = "blue") +
    labs(title = "Census Dataset - Actual Values\nOccurrences and degree of income\n Income<=50K (Red) Income>50K (Blue)",
         x = "Education", y = "Relationship",
         color = "% dots in Blue",
         size = "Occurrences") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)) 
	
print(base_income_plot)

### Accuracy and the Confusion Matrix

# The accuracy of every model will be calculated and compared. Accuracy is the percentage of correct Income predictions on the Census data set. 

# $$
# Accuracy = \frac{RightPredictions }{TotalPredictions}
# $$

# The confusion Matrix will also be reported for every model.  The confusion Matrix from R summarizes the performance of the model. It compares the predicted values against the train set values.

# To run the models and get the confusin matrix, two sets were created from the Census data set. The train set and the test set, which is 30% of the whole Census data set.

beep()

# Creating Train dataset and Test dataset. 

set.seed(42) 
test_index <- createDataPartition(census_data$income, times = 1, p = 0.7, list = FALSE) # create a 30% test set

censusTrain <- census_data[test_index,]
censusTest <- census_data[-test_index,]

# Following are the structure and number of rows that each of these datasets has:

beep()

cat("\ncensusTrain: Head\n")

head(censusTrain) %>% knitr::kable()

cat("\ncensusTrain: Number of Records\n")

tibble(dim(censusTrain))[1,1] %>% knitr::kable()

cat("\ncensusTest: Head\n")

head(censusTest) %>% knitr::kable()

cat("\ncensusTest: Number of Records\n")

tibble(dim(censusTest))[1,1] %>% knitr::kable()

rm(temp_test, temp_train)

### Models

# The following steps will be followed with all six classification models:

# 1. Predictions will be estimated
# 2. Plots will be generated
# 3. The confusion matrix, including Accuracy, will be produced and reported comparing the predictions on the train set to the actual values of the train set.

# Six models will be built and each model assessed against its Accuracy.
# The goal is to achieve more than 80% Accuracy and the model with the hightest Accuracy on the train set will be selected. Accuracy using the test set will be calculated to confirm the goal of greater than 80%.

# 1. KNN (K-Nearest Neighbors): It classifies data points based on their similarity to existing data points. The similarity is measured thorugh the euclidian distance and the minimum distance is chosen to predict the values. During the running, this model was saved in a file (train_knn.rds) and uploaded to GitHub.
# 2. KNN with Cross Validation Model: It is a KNN model that uses tunning to find the optimum value of K. During the running, this model was saved in a file (train_knn_cv.rds) and uploaded to GitHub.
# 3. Decision Tree: It splits data into subsets based on a series of conditions identified based on minimizing a coefficient. Each internal node represents a test (Yes or No) on a preditor, each branch represents the outcome of the test, and each leaf node represents the predicted value "<=50K"  or  ">50K". During the running, this model was saved in a file (train_dt.rds) and uploaded to GitHub.
# 4. Decision Tree - Complexity Parameter: The complexity parameter optimizes the size and complexity of the tree during growth and can improve generalization and avoid overfitting.  During the running, this model was saved in a file (train_dt_cp.rds) and uploaded to GitHub.
# 5. Random Forest: It randomly generates many decision trees from the same data set to strenghten the prediction.  During the running, this model was saved in a file (train_rf.rds) and uploaded to GitHub.
# 6. Random Forest with Cross Validation: It applies cross validation to the Random Forst model.  During the running, this model was saved in a file (train_rf_2.rds) and uploaded to GitHub.


######################################################################
#### First model: KNN\

# The first model applied is KNN (K-Nearest Neighbors), which classifies data points based on their similarity to existing data points. The similarity is measured through the euclidian distance and the minimum distance is chosen to select the neighborhood and predict the values. 

# During the first running, this model was saved in a file (train_knn.rds) and uploaded to GitHub. The model file is saved on the working directory and load it every time to save running time.

# The code to generate the model, save the model as a file and load the file with the model is as follows:

# KNN 

beep()

Sys.time()		# Time is shown to track duration

if (is.na(answerYes)) {
	# Code to execute if the user clicked on Cancel or Esc, The user has to go back and answer the question
	print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
	# Code to execute if the user answered "yes"
	print("Running from generated models downloaded from GitHub...")
	# Loading from file to save running time
	train_knn <- readRDS("train_knn.rds")
} else if (answerYes == FALSE) {
	# Code to execute if the user answered "no"
	print("Generating model...This may take minutes - hours")
	set.seed(6)
	train_knn <- 	train(income ~ .,
					method = "knn",
					data = censusTrain,
					tuneGrid = data.frame(k = seq(3, 23, 2)))
	# saveRDS(train_knn,"train_knn.rds")
} else {
  # Code to handle invalid input. this should not happen because NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

Sys.time()		# Time is shown to track duration

beep()

# The following is the resulting KNN model. First, it shows the  value of K that optimizes this model. Then, the model's characteristics.

train_knn$bestTune$k

train_knn

# train_knn$bestTune
# train_knn$finalModel

# Following is the Variables Importance plot for the KNN model. This graph shows how much each predictor variable contributes to the model's predictive power.\

# It shows that :\
# - relationship\
# - education\
# - age\

# Are the most important variables for this model.

# Next is a plot of Accuracy versus K showing that Accuracy is maximized at K =

train_knn$bestTune$k

plot(varImp(train_knn))

ggplot(train_knn, highlight = TRUE)

# The following section shows the Confusion Matrix for the Train set. Among other parameters, it shows:\

# - Accuracy
# - Sensitivity
# - Specificity 
# - Balanced Accuracy 

beep()

# Generation of predictions using KNN

# This step may take a few minutes
Sys.time()		# Time is shown to track duration
predicted_y_knn <- predict(train_knn, censusTrain)
Sys.time()		# Time is shown to track duration

Accuracy_knn <- confusionMatrix(predicted_y_knn, censusTrain$income)$overall[["Accuracy"]]

cat(paste("Accuracy: ",round(Accuracy_knn*100,2),"%"))

confusionMatrix(predicted_y_knn, censusTrain$income)

# The First model and its Accuracy are as follows:

beep()

Accuracy_results_t <- tibble(method = "1: KNN", Accuracy = Accuracy_knn, Parameter = paste("K = ",train_knn$bestTune$k))
Accuracy_results_t %>% knitr::kable()

########################################################################
#### Second model: KNN with Cross Validation\

# KNN with Cross Validation is a model that uses tunning to find the optimum value of K. 

# During the first running, this model was saved in a file (train_knn_cv.rds) and uploaded to GitHub. The model file is saved on the working directory and load it every time to save running time.

# The code to generate the model, save the model as a file and load the file with the model is as follows:

# KNN Cross Validation ==========================================

beep()

Sys.time()		# Time is shown to track duration

if (is.na(answerYes)) {
	# Code to execute if the user clicked on Cancel or Esc, The user has to go back and answer the question
	print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
	# Code to execute if the user answered "yes"
	print("Running from generated models downloaded from GitHub...")
	# Loading from file to save running time
	train_knn_cv <- readRDS("train_knn_cv.rds")
} else if (answerYes == FALSE) {
	# Code to execute if the user answered "no"
	print("Generating model...This may take minutes - hours")
	control <- trainControl(method = "cv", number = 5, p = .9)
	set.seed(6)
	train_knn_cv <- train(income ~ ., method = "knn",
						data = censusTrain,
						tuneGrid = data.frame(k = seq(3, 31, 2)),
						trControl = control)
	# saveRDS(train_knn_cv,"train_knn_cv.rds")
} else {
  # Code to handle invalid input. this should not happen because NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

Sys.time()		# Time is shown to track duration

beep()

# The following is the resulting KNN CV model. First, it shows the  value of K that optimizes this model. Then, the model's characteristics.

train_knn_cv$bestTune$k

train_knn_cv

# train_knn_cv$bestTune
# train_knn_cv$finalModel

# Following is the Variables Importance plot for the KNN_CV model. This graph shows how much each predictor variable contributes to the model's predictive power.\

# It shows that :\
# - relationship\
# - education\
# - age\

# Are the most important variables for this model.

# Variables Importance

plot(varImp(train_knn_cv))

# Next is a plot of Accuracy versus K showing that Accuracy is maximized at K =
train_knn_cv$bestTune$k

ggplot(train_knn_cv, highlight = TRUE)

# The following plot is illustrative of the behavior of the KNN CV model. Because there are 9 predictors, they cannot be plotted in just one graph with two dimensions. A reduced KNN CV model was generated with just the two most important CONTINUOUS variables for KNN:  "Age" and "Hours per Week". The contour of the reduced model's predictions was then plotted against CensusTrain. The contour shows the decisions' boundaries to assign "<=50K" or ">50K"

beep()

# Generation of predictions usingn KNN CV

# Sys.time()		# Time is shown to track duration
predicted_y_knn_cv <- predict(train_knn_cv, censusTrain)
# Sys.time()		# Time is shown to track duration

# dev.off()    # run before plotting KNN if issues with this graph - to clean plotting

# Plotting KNN most important continuous variables versus income

# Define the grid limits
x_min <- min(censusTrain$age) - 1
x_max <- max(censusTrain$age) + 1
y_min <- min(censusTrain$hoursperweek) - 1
y_max <- max(censusTrain$hoursperweek) + 1

# Create a grid of values
grid_income <- expand.grid(age = seq(x_min, x_max, by = 5),
                    hoursperweek = seq(y_min, y_max, by = 5))
					
k <- train_knn_cv$bestTune$k  # Number of neighbors
df_income <- data.frame(censusTrain[, 1], censusTrain[, 9])

predicted_income <- knn(train = df_income, test = grid_income, 
                         cl = censusTrain$income, k = k)
			
# Add predictions to the grid data
grid_income$PredictedIncome <- predicted_income

# Base plot with original data points
base_income_plot <- ggplot(censusTrain, aes(x = age, y = hoursperweek, 
                                     color = income)) +
  geom_point(size = 2) +
  labs(title = "Reduced k-NN(age, hoursperweek) Decisions' Boundaries",
       x = "Age",
       y = "Hours per Week") +
  theme_minimal()
  
# print(base_income_plot)

# Add decision boundaries using geom_contour
contour_income_plot <- base_income_plot +
  geom_contour(data = grid_income, aes(x = age, y = hoursperweek, 
                                z = as.numeric(PredictedIncome)),
               bins = 2, color = "black", linewidth=1) +
  scale_fill_manual(values = c("red", "green", "blue"))

# Display the plot
print(contour_income_plot)

# The following section shows the Confusion Matrix for the Train set. Among other parameters, it shows:\

# - Accuracy
# - Sensitivity
# - Specificity 
# - Balanced Accuracy 

Accuracy_knn_cv <- confusionMatrix(predicted_y_knn_cv, censusTrain$income)$overall[["Accuracy"]]

cat(paste("Accuracy: ",round(Accuracy_knn_cv*100,2),"%"))

confusionMatrix(predicted_y_knn_cv, censusTrain$income)

# The second model and its Accuracy are as follows:

beep()

# The second model and its Accuracy are as follows:

Accuracy_results_t <- bind_rows(Accuracy_results_t,
                          tibble(method="2: KNN with CV (Cross Validation)", 
						  Accuracy = Accuracy_knn_cv, 
						  Parameter = paste("K = ",train_knn_cv$bestTune$k)))
Accuracy_results_t %>% knitr::kable()



#### Third model: Decision Tree\


# Decision Tree splits data into subsets based on a series of conditions identified based on minimizing a coefficient. Each internal node represents a test (Yes or No) on a predictor, each branch represents the outcome of the test, and each leaf node represents the predicted value "<=50K"  or  ">50K". 

# During the first running, this model was saved in a file (train_dt.rds) and uploaded to GitHub. The model file is saved on the working directory and load it every time to save running time.

# The code to generate the model, save the model as a file and load the file with the model is as follows:


# DECISION TREE ==========================================================

beep()

Sys.time()		# Time is shown to track duration

if (is.na(answerYes)) {
	# Code to execute if the user clicked on Cancel or Esc, The user has to go back and answer the question
	print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
	# Code to execute if the user answered "yes"
	print("Running from generated models downloaded from GitHub...")
	# Loading from file to save running time
	train_dt <- readRDS("train_dt.rds")
} else if (answerYes == FALSE) {
	# Code to execute if the user answered "no"
	print("Generating model...This may take minutes - hours")
	set.seed(6)
	train_dt <- rpart(income ~ ., data = censusTrain)
	# saveRDS(train_dt,"train_dt.rds")
} else {
  # Code to handle invalid input. this should not happen because NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

Sys.time()		# Time is shown to track duration

# Next is the plot of the Decision Tree model followed by the rules of the model and its Accuracy.\

beep()

# display the decision tree

rpart.plot(x= train_dt, type= 5, extra = 0,tweak = 1.2)

# display model

train_dt

# Following is the Variables Importance list for the Decision Tree model. This list shows how much each predictor variable contributes to the model's predictive power.\

# It shows that :\
# - relationship\
# - maritalstatus\
# - education\

# Are the most important variables for this model.

# Variables importance

listVarImp<-as.data.frame(varImp(train_dt))
listVarImp<-arrange(listVarImp, desc(Overall))
listVarImp %>% knitr::kable()


# The following section shows the Confusion Matrix for the Train set. Among other parameters, it shows:\

# - Accuracy
# - Sensitivity
# - Specificity 
# - Balanced Accuracy 
 
# Generation of predictions using Decision Tree

beep()

# Sys.time()		# Time is shown to track duration
predicted_y_dt <- predict(train_dt, censusTrain, type = "class")
# Sys.time()		# Time is shown to track duration

Accuracy_dt <- confusionMatrix(predicted_y_dt, censusTrain$income)$overall[["Accuracy"]]

Accuracy_dt

confusionMatrix(predicted_y_dt, censusTrain$income)


# The third model and its Accuracy are as follows:

beep()

# The third model and its Accuracy are as follows:

Accuracy_results_t <- bind_rows(Accuracy_results_t,
                          tibble(method="3: Decision tree", 
						  Accuracy = Accuracy_dt, 
						  Parameter = paste(" ")))
Accuracy_results_t %>% knitr::kable()


#### Fourth model: Decision Tree with Complexity Parameter\

# The complexity parameter in a Decision Tree optimizes the size and complexity of the tree during growth and can improve generalization and avoid overfitting.

# During the first running, this model was saved in a file (train_dt_cp.rds) and uploaded to GitHub. The model file is saved on the working directory and load it every time to save running time.

# The code to generate the model, save the model as a file and load the file with the model is as follows:

# Decision Tree with Complexity Parameter (CP) =========================================

beep()

Sys.time()		# Time is shown to track duration

if (is.na(answerYes)) {
	# Code to execute if the user clicked on Cancel or Esc, The user has to go back and answer the question
	print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
	# Code to execute if the user answered "yes"
	print("Running from generated models downloaded from GitHub...")
	# Loading from file to save running time
	train_dt_cp <- readRDS("train_dt_cp.rds")
} else if (answerYes == FALSE) {
	# Code to execute if the user answered "no"
	print("Generating model...This may take minutes - hours")
	set.seed(6)
	train_dt_cp <- train(income ~ .,
						method = "rpart",
						tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
						data = censusTrain)
	# saveRDS(train_dt_cp,"train_dt_cp.rds")
} else {
  # Code to handle invalid input. this should not happen because NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

Sys.time()		# Time is shown to track duration

# Next is the Decision Tree model and the plot of Complexity Parameter versus Accuracy. It shows that Accuracy is maximum for a CP of:

train_dt_cp$bestTune$cp

beep()

train_dt_cp

plot(train_dt_cp)

# Following are the plot of the Decision Tree model with Complexity Parameter and the description of its rules.\

# Plotting decision tree

rpart.plot(x= train_dt_cp$finalModel)
# rpart.plot(x= train_dt_cp$finalModel, type= 5, extra = 0,tweak = 1.2)

# train_dt_cp$bestTune
# train_dt_cp$bestTune$cp

train_dt_cp$finalModel

# The Variables and Values Importance plot is shown for the Decision Tree model. This plot shows how much each variable and specific values contribute to the model's predictive power.\

# Variables and Values importance

# varImp(train_dt_cp$finalModel)
# varImp(train_dt_cp)
plot(varImp(train_dt_cp), top=15)

# The following section shows the Confusion Matrix for the Train set. Among other parameters, it shows:\

# - Accuracy
# - Sensitivity
# - Specificity 
# - Balanced Accuracy 

# Generation of predictions using Decision Tree with Complexity Parameter

beep()

# Sys.time()		# Time is shown to track duration
predicted_y_dt_cp <- predict(train_dt_cp, censusTrain)
# Sys.time()		# Time is shown to track duration

# compute accuracy
Accuracy_dt_cp <- confusionMatrix(factor(predicted_y_dt_cp), factor(censusTrain$income))$overall["Accuracy"]

Accuracy_dt_cp

confusionMatrix(factor(predicted_y_dt_cp), factor(censusTrain$income))

# The fourth model and its Accuracy are as follows:

beep()

# The fourth model and its Accuracy are as follows:

Accuracy_results_t <- bind_rows(Accuracy_results_t,
                          tibble(method="4: Decision tree with CP (Complexity Parameter)", 
						  Accuracy = Accuracy_dt_cp, 
						  Parameter = paste("cp = ", train_dt_cp$bestTune$cp)))
Accuracy_results_t %>% knitr::kable()


#### Fifth model: Random Forest\


# Random Forest generates many decision trees from the same data set to strenghten the prediction.  

# During the first running, this model was saved in a file (train_rf.rds) and uploaded to GitHub. The model file is saved on the working directory and load it every time to save running time.

# The code to generate the model, save the model as a file and load the file with the model is as follows:

# Random Forest =========================================

beep()

Sys.time()		# Time is shown to track duration

# For the generation of this report, the randomForest function was used

if (is.na(answerYes)) {
	# Code to execute if the user clicked on Cancel or Esc, The user has to go back and answer the question
	print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
	# Code to execute if the user answered "yes"
	print("Running from generated models downloaded from GitHub...")
	# Loading from file to save running time
	train_rf <- readRDS("train_rf.rds")
} else if (answerYes == FALSE) {
	# Code to execute if the user answered "no"
	print("Generating model...This may take minutes - hours")
	set.seed(6)
	train_rf<-randomForest(formula=income~.,
								data=censusTrain,
	                            ntree=500,
								do.trace=TRUE)
	# saveRDS(train_rf,"train_rf.rds")
} else {
  # Code to handle invalid input. this should not happen because NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

Sys.time()		# Time is shown to track duration

beep()

# Next is the Random Forest model using 500 trees with 3 variables tried at each split.


print(train_rf) # Print the model information

# The variables Importance plot is shown for the Random Forest model. This plot shows how much each variable contributes to the models's predictive power. It shows that Age, Relationship and Education are the most important variables for the model.
# This is also reflected in the plot of Variables versus MeanDecreaseGini that follows.

# Variables importance

listVarImp<-as.data.frame(varImp(train_rf))
listVarImp<-arrange(listVarImp, desc(Overall))
listVarImp %>% knitr::kable()

varImpPlot(train_rf)

plot(train_rf, log="y")

# The previous plot shows how error is decreasing as number of trees is increasing

# The following two graphs show the dependance of the model on the values of the two most important variables: age and relationship.

# Importance of specific variables

partialPlot(train_rf, censusTrain, age, "<=50K")

partialPlot(train_rf, censusTrain, relationship, "<=50K", las = 3, xlab = "")

# It shows that the model is more dependant for Ages less than 25 and Relationship as Husband and Wife.

# The following section shows the Confusion Matrix for the Train set. Among other parameters, it shows:\

# - Accuracy
# - Sensitivity
# - Specificity 
# - Balanced Accuracy 

# Generation of predictions using Random Forest

beep()

# Sys.time()		# Time is shown to track duration
predicted_y_rf <- predict(train_rf, censusTrain)
# Sys.time()		# Time is shown to track duration

Accuracy_rf <- confusionMatrix(factor(predicted_y_rf), factor(censusTrain$income))$overall["Accuracy"]

Accuracy_rf

confusionMatrix(factor(predicted_y_rf), factor(censusTrain$income))

# The fifth model and its Accuracy are as follows:

beep()

# The fifth model and its Accuracy are as follows:

Accuracy_results_t <- bind_rows(Accuracy_results_t,
                          tibble(method="5: Random forest", 
						  Accuracy = Accuracy_rf, 
						  Parameter = paste("n of trees = 500, n of var = 3")))
Accuracy_results_t %>% knitr::kable()


#### Sixth model: Random Forest with Cross Validation\

# Cross validation is applied to the Random Forest model.  

# During the first running, this model was saved in a file (train_rf_2.rds) and uploaded to GitHub. The model file is saved on the working directory and load it every time to save running time.

# The code to generate the model, save the model as a file and load the file with the model is as follows:

# Random Forest with CV =========================================

beep()

Sys.time()		# Time is shown to track duration

if (is.na(answerYes)) {
	# Code to execute if the user clicked on Cancel or Esc, The user has to go back and answer the question
	print("Go back to run and answer Yes or No to the question.")
} else if (answerYes == TRUE) {
	# Code to execute if the user answered "yes"
	print("Running from generated models downloaded from GitHub...")
	# Loading from file to save running time
	train_rf_2 <- readRDS("train_rf_2.rds")
} else if (answerYes == FALSE) {
	# Code to execute if the user answered "no"
	print("Generating model...This may take minutes - hours")
	set.seed(6)
	train_rf_2 <- train(income ~ .,
						method = "Rborist",
						tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
						data = censusTrain)
	# saveRDS(train_rf_2,"train_rf_2.rds")
} else {
  # Code to handle invalid input. this should not happen because NA is the first IF
  print("Go back to run and answer Yes or No to the question.")
}

Sys.time()		# Time is shown to track duration

beep()

# Next is the Random Forest model with Cross Validation generated using 300 trees with 3 variables tried at each split.

print(train_rf_2) # Print the model information

# The Variables and Values Importance plot is shown for the Random Forest CV model. This plot shows how much each variable and Values contribute to the model's predictive power.

# It shows that: 

# - MaritalStatus-Married-Civ-Spouse
# - Age
# - MaritalStatus-Never-married

# Are the most important variables and values for this model.

# Variables importance
plot(varImp(train_rf_2), top = 15)

plot(train_rf_2, log="y")

# The previous plot shows how Accuracy decreases as Minimal Node Size increases.

# The following section shows the Confusion Matrix for the Train set. Among other parameters, it shows:\

# - Accuracy
# - Sensitivity
# - Specificity 
# - Balanced Accuracy 

# Generation of predictions using Random Forest with Cross Validation

beep()

# Sys.time()		# Time is shown to track duration
predicted_y_rf_CV <- predict(train_rf_2, censusTrain)
# Sys.time()		# Time is shown to track duration

Accuracy_rf_CV <- confusionMatrix(factor(predicted_y_rf_CV), factor(censusTrain$income))$overall["Accuracy"]

Accuracy_rf_CV

confusionMatrix(factor(predicted_y_rf_CV), factor(censusTrain$income))

# The sixth model and its Accuracy are as follows:

beep()

# The sixth model and its Accuracy are as follows:

Accuracy_results_t <- bind_rows(Accuracy_results_t,
                          tibble(method="6: Random forest with CV (Cross Valdidation)", 
						  Accuracy = Accuracy_rf_CV, 
						  Parameter = paste("")))
Accuracy_results_t %>% knitr::kable()


# RESULTS


# The following table shows all six models and their Accuracies:

Accuracy_results_t %>% knitr::kable()

# Accuracy has been improving when incorporating Cross Validation or the Complexity Parameter to the base models.

# The Random Forest model optimized Accuracy among the six models.

# With the Random Forest model, an Income prediction model was built with an Accuracy on the Train dataset of:

cat(paste("Accuracy: ",round(Accuracy_rf*100,2),"%"))

# Now, the selected model (Random Forest) is applied to the Test dataset and its accuracy calculated

Sys.time()		# Time is shown to track duration
predicted_y_rf_test <- predict(train_rf, censusTest)
Sys.time()		# Time is shown to track duration

# compute accuracy
Accuracy_rf_test <- confusionMatrix(factor(predicted_y_rf_test), factor(censusTest$income))$overall["Accuracy"]

cat(paste(round(Accuracy_rf_test*100,2),"%"))

confusionMatrix(factor(predicted_y_rf_test), factor(censusTest$income))

# The selected model and its Accuracy are as follows:

beep()

Accuracy_results_t <- bind_rows(Accuracy_results_t,
                          tibble(method="Random Forest (Test dataset)", 
						  Accuracy = Accuracy_rf_test, 
						  Parameter = paste("n of trees = 500, n of var = 3")))
Accuracy_results_t %>% knitr::kable()

# The goal of having 80+ percent accuracy has been confirmed with the test dataset.

# The following graph shows the actual values of Income for the  Census Test dataset. Because this is a two dimensional graph, the graph is illustrative and only two predictors out of the nine are shown, Education and Relationship.

# Every dot shows the frequency of occurrences that have ">50K" as income in blue as a percentage of the total number of records for that combination of Education and Relationship. The size of the dot represents the number of records for that combination of Education and Relationship.

#  x = "Education", y = "Relationship"

df_income <- data.frame(censusTest[, 3], censusTest[, 6], censusTest[,10])
colnames(df_income) <- c("Education", "Relationship", "Income")

# print(df_income)

# Summarize the count of value and total count per group
proportion_df <- df_income %>%
  group_by(Relationship, Education) %>%
  summarise(
    count_filtered = sum(Income == ">50K"),
    total_count = n(),
    Red_to_Blue = count_filtered / total_count,
    .groups = "drop"
  )

# print(proportion_df)

base_income_plot <- ggplot(proportion_df, aes(x = Education, y = Relationship, 
                          color = Red_to_Blue, 
                          size = total_count)) +
    geom_point() +
    scale_color_gradient(low = "red", high = "blue") +
    labs(title = "Census Test Actual Values (Red:  <=50K    Blue: >50K)",
         color = "% dots in Blue",
         size = "Occurrences") +
  	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)) 
#	theme(axis.text.x = element_blank(), legend.position = "none")

print(base_income_plot)



# The following graph grid shows an illustrative visual comparison between the actual values of the census test dataset and a graph generated with the Test predicted values of the model with the hightest Accuracy (Random Forest). This is done with just two predicotrs so it can be shown in a graph.

#  x = "Education", y = "Relationship"

base_income_plot <- ggplot(proportion_df, aes(x = Education, y = Relationship, 
                          color = Red_to_Blue, 
                          size = total_count)) +
    geom_point() +
    scale_color_gradient(low = "red", high = "blue") +
    labs(title = "Census Test (Actual values)",
         x = "", y = "",
         color = "% dots in Blue",
         size = "Occurrences") +
  	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none") 
#	theme(axis.text.x = element_blank(), legend.position = "none")

# print(base_income_plot)

df_income_prediction <- data.frame(censusTest[, 3], censusTest[, 6], predicted_y_rf_test)
colnames(df_income_prediction) <- c("Education", "Relationship", "Income")

# Summarize the count of value and total count per group
proportion_df_prediction <- df_income_prediction %>%
  group_by(Relationship, Education) %>%
  summarise(
    count_filtered = sum(Income == ">50K"),
    total_count = n(),
    Red_to_Blue = count_filtered / total_count,
    .groups = "drop"
  )

predicted_y_rf_test_plot <- ggplot(proportion_df_prediction, aes(x = Education, y = Relationship, 
                          color = Red_to_Blue, 
                          size = total_count)) +
    geom_point() +
    scale_color_gradient(low = "red", high = "blue") +
    labs(title = "Random Forest (Predictions)",
         x = "", y = "",
         color = "% dots in Blue",
         size = "Occurrences") +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), legend.position = "none") 
#	theme(axis.text.x = element_blank(), legend.position = "none") 

# print(predicted_y_rf_test_plot)

grid.arrange(	
				base_income_plot, 
				predicted_y_rf_test_plot, 
				ncol = 2, 
				top="Census Test values and Random Forest predictions on Income", bottom="", 
				left="", right="")	
				
grid.rect(gp=gpar(fill=NA))


# CONCLUSION


# In this analysis, six models were considered and accuracy was used to measure the predictive power of each model.


# The models were increasing accuracy to get to the maximum with the Random Forest  model.

# Using Random Forest, an Income prediction model was built with an Accuracy on the test set of:

cat(paste(round(Accuracy_rf*100,2),"%"))

# One potential limitation of the analysis is that the dataset has few records with income >50K in comparison to records with Income <=50K. This maybe a problem of Prevalence (when the ratio of Trues is close to 0 or 1).

# A couple of ways to address the issue of prevalence are:

# - generating a sample without prevalence
# - analyzing sensitivity and specificity


# An additional step that may improve the prediction is to ensemble all six models into one. Ensembles combine multiple machine learning algorithms into one model to improve predictions (2).

# REFERENCES

#  1. Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
 
 # 2.  Irizarry, R. Introduction to Data Science  - Data Analysis and Prediction Algorithms with R (32.5 Ensembles)
#  https://rafalab.dfci.harvard.edu/dsbook/machine-learning-in-practice.html#ensembles
 
 


