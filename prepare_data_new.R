if (! "RTextTools" %in% row.names(installed.packages()))
  install.packages("RTextTools")
library(RTextTools)
if (! "tm" %in% row.names(installed.packages()))
  install.packages("tm")
library(tm)
if (! "rpart" %in% row.names(installed.packages()))
  install.packages("rpart")
library(rpart)

if (! "rpart.plot" %in% row.names(installed.packages()))
  install.packages("rpart.plot")
library(rpart.plot)
if (! "maptree" %in% row.names(installed.packages()))
  install.packages("maptree")
library(maptree)
if (! "CORElearn" %in% row.names(installed.packages()))
  install.packages("CORElearn")
library(CORElearn)

# confusionMatrix - TP rate, itp.
if (! "caret" %in% row.names(installed.packages()))
  install.packages("caret")
library(caret)

if (! "e1071" %in% row.names(installed.packages()))
  install.packages("e1071")
library(e1071)

if (! "randomForest" %in% row.names(installed.packages()))
  install.packages("randomForest")
library(randomForest)

source("naive_bayes_experiments.R")
source("dec_tree_experiment.R")
source("svm_experiments.R")



spam.dir <- "./spamassasin/"

get.msg <- function(path.dir) {
  con <- file(path.dir, open="rt", encoding = "latin1")
#  print("con:\n\n")
#  print(con)
#  print("\n")
  
  text <- readLines(con)
#  print("text:\n\n")
#  print(text)
#  print("\n")  
  
  msg <- text[seq(which(text=="")[1]+1, length(text), 1)]
#  print("msg:\n\n")
#  print(msg)
#  print("\n")
  
  close(con)
  return (paste(msg, collapse="\n"))
}

get.all <- function(path.dir) {
  all.file <- dir(path.dir)
  all.file <- all.file[which(all.file!="cmds")]
  msg.all <- sapply(all.file, function(p) get.msg(paste0(path.dir, p)))
}

easy_ham.all <- get.all(paste0(spam.dir,"easy_ham/"))
easy_ham_2.all <- get.all(paste0(spam.dir, "easy_ham_2/"))

hard_ham.all <- get.all(paste0(spam.dir, "hard_ham/"))

spam.all <- get.all(paste0(spam.dir, "spam/"))
spam_2.all <- get.all(paste0(spam.dir, "spam_2/"))

# Reprezentacja danych jako Data Frame
easy_ham.df <- as.data.frame(easy_ham.all, stringsAsFactors = FALSE)
easy_ham_2.df <- as.data.frame(easy_ham_2.all, stringsAsFactors = FALSE)

hard_ham.df <- as.data.frame(hard_ham.all, stringsAsFactors = FALSE)

spam.df <- as.data.frame(spam.all, stringsAsFactors = FALSE)
spam_2.df <- as.data.frame(spam_2.all, stringsAsFactors = FALSE)

rownames(easy_ham.df) <- NULL
rownames(easy_ham_2.df) <- NULL
rownames(hard_ham.df) <- NULL
rownames(spam_2.df) <- NULL
rownames(spam.df) <- NULL

# Klasa: 0 - ham, 1 - spam

easy_ham.df$email_class <- factor("ham")
easy_ham_2.df$email_class <- factor("ham")
hard_ham.df$email_class <- factor("ham")
spam.df$email_class <- factor("spam")
spam_2.df$email_class <- factor("spam")

names(easy_ham.df) <- c("text", "email_class")
names(easy_ham_2.df) <- c("text", "email_class")
names(hard_ham.df) <- c("text", "email_class")
names(spam_2.df) <- c("text", "email_class")
names(spam.df) <- c("text", "email_class")

email.data <- rbind(easy_ham.df, hard_ham.df, spam.df,
                    easy_ham_2.df, spam_2.df)
email.data <- email.data[sample(nrow(email.data)),]
## 80% of the sample size
train_size <- floor(0.80 * nrow(email.data))

## set the seed to make your partition reproducible
set.seed(12345)
train_ind <- sample(seq_len(nrow(email.data)), size = train_size)

names(email.data) <- c("text", "email_class")

removeHTMLAttr <- function(x) {
  x <- gsub(pattern = "(<\\w+)[^>]*(>)", "\\1\\2", x) # usuwanie atrybutow w tagach HTML
  return(x)
}
removeHTMLClosingTag <- function(x) {
  x <- gsub(pattern = "</[^>]+>", "", x)
}
isContainingHTMLTags <- function(x) {
  return(grepl("<[^>]+>", x))
}

for(i in 1: nrow(email.data)) {
  email.data$text[i] <- removeHTMLClosingTag(removeHTMLAttr(train.test$text[i]))
  email.data$isContainingHTML[i] <- isContainingHTMLTags(train.test$text[i])
}

#dtm.tfidf.test <- create_matrix(train.test$text, language = "english", minWordLength = 3, removeNumbers = TRUE, stemWords = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.95, weighting = weightTfIdf)
##############################################
# dane do modeli predykcyjnych - TFIDF

dtm.tfidf <- create_matrix(email.data$text, 
                           language = "english", 
                           minWordLength = 3,
                           toLower = TRUE,
                           removeStopwords = TRUE,
                           removeNumbers = TRUE,
                           stemWords = TRUE,
                           removePunctuation = TRUE,
                           removeSparseTerms = 0.99, 
                           weighting = weightTfIdf
)
data.tfidf <- cbind(email.data, as.matrix(dtm.tfidf))
names(data.tfidf) <- make.names(names(data.tfidf)) 

train.tfidf.DF <- data.tfidf[train_ind,]
test.tfidf.DF <- data.tfidf[-train_ind,]
data.tfidf$text <- NULL

train.tfidf.DF$text = NULL
test.tfidf.DF$text = NULL
###########################################
# dane do modeli predykcyjnych - TF
dtm.tf <- create_matrix(email.data$text, 
                        language = "english", 
                        minWordLength = 3,
                        toLower = TRUE,
                        removeStopwords = TRUE,
                        removeNumbers = TRUE,
                        stemWords = TRUE,
                        removePunctuation = TRUE,
                        removeSparseTerms = 0.99,
                        weighting = weightTf
)
data.tf <- cbind(email.data, as.matrix(dtm.tf))
names(data.tf) <- make.names(names(data.tf)) 

train.tf.DF <- data.tf[train_ind,]
test.tf.DF <- data.tf[-train_ind,]

train.tf.DF$text = NULL
test.tf.DF$text = NULL
data.tf$text <- NULL

########################################
# binarna reprezentacja - 0 lub 1
dtm.bin <- create_matrix(email.data$text,
                        language = "english",
                        minWordLength = 3,
                        toLower = TRUE,
                        removeStopwords = TRUE,
                        removeNumbers = TRUE,
                        stemWords = TRUE,
                        removePunctuation = TRUE,
                        removeSparseTerms = 0.99,
                        weighting = weightBin
)
data.bin <- cbind(email.data, as.matrix(dtm.bin))
names(data.bin) <- make.names(names(data.bin)) 

train.bin.DF <- data.bin[train_ind,]
test.bin.DF <- data.bin[-train_ind,]

train.bin.DF$text = NULL
test.bin.DF$text = NULL
data.bin$text <- NULL

############################################

# wybór liczby atrybutów
varImportance.tfidf <- attrEval(email_class~., data.tfidf, estimator="Gini")
varImportance.tfidf <- data.frame(varImportance.tfidf, names(varImportance.tfidf))
names(varImportance.tfidf) <- c("importance", "term")
varImportance.tfidf.decreasing <- varImportance.tfidf[order(varImportance.tfidf$importance, decreasing = TRUE),]


plot(1:nrow(varImportance.tfidf.decreasing),
     varImportance.tfidf.decreasing[, 1],
     xlab='Numery cech, uporządkowanych według istotności malejąco',
     ylab='Istotność')
features_num <- 200

varImportance.tfidf.decreasing.columns <- order(varImportance.tfidf$importance, decreasing = TRUE)[1:features_num]
data.tfidf.selected_features <- data.tfidf[, varImportance.tfidf.decreasing.columns[1:features_num] + 1]
data.tfidf.selected_features$email_class <- data.tfidf$email_class

#TODO: remove train & test after tests
train.tfidf.DF.selected_features <- train.tfidf.DF[, varImportance.tfidf.decreasing.columns[1:features_num] + 1]
train.tfidf.DF.selected_features$email_class = train.tfidf.DF$email_class
test.tfidf.DF.selected_features <- test.tfidf.DF[, varImportance.tfidf.decreasing.columns[1:features_num] + 1]
test.tfidf.DF.selected_features$email_class = test.tfidf.DF$email_class


varImportance.tf <- attrEval(email_class~., data.tf, estimator="Gini")
varImportance.tf <- data.frame(varImportance.tf, names(varImportance.tf))
names(varImportance.tf) <- c("importance", "term")
varImportance.tf.decreasing <- varImportance.tf[order(varImportance.tf$importance, decreasing = TRUE),]
varImportance.tf.decreasing.columns <- order(varImportance.tf$importance, decreasing = TRUE)[1:features_num]
data.tf.selected_features <- data.tf[, varImportance.tf.decreasing.columns[1:features_num] + 1]
data.tf.selected_features$email_class <- data.tf$email_class
#TODO: remove train & test after tests

train.tf.DF.selected_features <- train.tf.DF[, varImportance.tf.decreasing.columns[1:features_num] + 1]
train.tf.DF.selected_features$email_class = train.tf.DF$email_class
test.tf.DF.selected_features <- test.tf.DF[, varImportance.tf.decreasing.columns[1:features_num] + 1]
test.tf.DF.selected_features$email_class = test.tf.DF$email_class


varImportance.bin <- attrEval(email_class~., data.bin, estimator="Gini")
varImportance.bin <- data.frame(varImportance.bin, names(varImportance.bin))
names(varImportance.bin) <- c("importance", "term")
varImportance.bin.decreasing <- varImportance.bin[order(varImportance.bin$importance, decreasing = TRUE),]
varImportance.bin.decreasing.columns <- order(varImportance.bin$importance, decreasing = TRUE)[1:features_num]
data.bin.selected_features <- data.bin[, varImportance.bin.decreasing.columns[1:features_num] + 1]
data.bin.selected_features$email_class <- data.bin$email_class

#TODO: remove train & test after tests

train.bin.DF.selected_features <- train.bin.DF[, varImportance.bin.decreasing.columns[1:features_num] + 1]
train.bin.DF.selected_features$email_class = train.bin.DF$email_class
test.bin.DF.selected_features <- test.bin.DF[, varImportance.bin.decreasing.columns[1:features_num] + 1]
test.bin.DF.selected_features$email_class = test.bin.DF$email_class
###############################
# testy naiwnego klasyfikatora Bayesa
bayes_results_tf_idf <- naive_bayes_misclassification_costs_tests(5, 
                                                                  data.tfidf.selected_features, 
                                                                  c(0.01, 0.1, 1, 10, 100))
bayes_results_tf <- naive_bayes_misclassification_costs_tests(5, 
                                                              data.tf.selected_features, 
                                                              c(0.01, 0.1, 1, 10, 100))
bayes_results_bin <- naive_bayes_misclassification_costs_tests(5, 
                                                              data.bin.selected_features, 
                                                              c(0.01, 0.1, 1, 10, 100))
bayes_y_range <- range(bayes_results_bin[, 8], 
                       bayes_results_tf[, 8],
                       bayes_results_tf_idf[, 8])
plot(bayes_results_tf_idf[, 1], bayes_results_tf_idf[, 8], log='x', col='red', 
     ylim=bayes_y_range, ylab = 'Błąd klasyfikacji',
     xlab=expression(frac("Koszt błędnej klasyfikacji spamu", "Koszt błędnej klasyfikacji pożądanej korespondencji")),
     mgp = c(3, 0.1, 0))
points(bayes_results_tf[, 1], bayes_results_tf[, 8], col='green')
points(bayes_results_bin[, 1], bayes_results_bin[, 8], col='black')
legend('center', legend=c("tf-idf", "tf", "binarna"),
      col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
      title="Reprezentacja tekstu")

##############################3
#TESTY DRZEW DECYZYJNYCH
tree_results_tf_idf <- decision_tree_grid_search_tests(5, data.tfidf.selected_features, 
                                                       c(1, 5, 10, 15), 
                                                       c(1, 5, 10, 25, 50))
tree_results_tf <- decision_tree_grid_search_tests(5, data.tf.selected_features, 
                                                       c(1, 5, 10, 15), 
                                                       c(1, 5, 10, 25, 50))
tree_results_bin <- decision_tree_grid_search_tests(5, data.bin.selected_features, 
                                                       c(1, 5, 10, 15), 
                                                       c(1, 5, 10, 25, 50))


############### SVM ###############
results_svm_tf_idf_gammas <- svm_rbf_gamma_tests(5, data.tfidf.selected_features,
                                                 c(0.001, 0.005, 0.01, 0.025, 0.05))

results_svm_tf_gammas <- svm_rbf_gamma_tests(5, data.tf.selected_features,
                                                 c(0.001, 0.005, 0.01, 0.025, 0.05))

results_svm_bin_gammas <- svm_rbf_gamma_tests(5, data.bin.selected_features,
                                                 c(0.001, 0.005, 0.01, 0.025, 0.05))


results_svm_tf_idf_polynomial <- svm_polynomial_degree_tests(5, data.tfidf.selected_features,
                                           c(1, 2, 3))

results_svm_tf_polynomial <- svm_polynomial_degree_tests(5, data.tf.selected_features,
                                                             c(1, 2, 3))

results_svm_bin_polynomial <- svm_polynomial_degree_tests(5, data.bin.selected_features,
                                                             c(1, 2, 3))

results_svm_tf_idf_rbf_cost <- svm_cost_rbf_tests(5, data.tfidf.selected_features, 0.005,
                                              c(0.1, 1, 10, 100))

results_svm_tf_rbf_cost <- svm_cost_rbf_tests(5, data.tf.selected_features, 0.005,
                                                  c(0.1, 1, 10, 100))

results_svm_bin_rbf_cost <- svm_cost_rbf_tests(5, data.bin.selected_features, 0.005,
                                                  c(0.1, 1, 10, 100))










#############################################
# model po selekcji 25 atrybutów z randomForest
words <- rownames(varImportance.tfidf.decreasing)[1:features_num]
fmla <- as.formula(paste("email_class ~ ", paste(words, collapse = "+")))

model.tree.25 <- rpart(fmla, data=train.tfidf.DF.selected_features,
                       minsplit = 5, maxdepth = 5)
prp(model.tree.25)
pred.tree.25 <- predict(model.tree.25, test.tfidf.DF.selected_features,
                        type="class")
mytable <-table(test.tfidf.DF$email_class, pred.tree.25, dnn=c("Obs", "Pred"))


#classification_error <- sum(pred.tree.25 != test.tfidf.DF$email_class) / NROW(pred.tree.25)

model.tree.cp <- rpart(email_class~., data=train.tfidf.DF, cp = 0.02, minbucket = 30)
pred.tree.cp <- predict(model.tree.cp, test.tfidf.DF, type = "email_class")
table(test.tfidf.DF$email_class, pred.tree.cp, dnn=c("Obs", "Pred"))
prp(model.tree.cp)

results <- decision_tree_grid_search_tests(5, data.tfidf.selected_features, words, c(5), 
                                           c(5))

# klasyfikator Bayesa
model.bayes <- naiveBayes(email_class~., data = train.tfidf.DF.selected_features, laplace = 3)
pred.bayes <- predict(model.bayes, test.tfidf.DF.selected_features, type = "raw")
table(pred = pred.bayes, true = test.tfidf.DF.selected_features$email_class, dnn=c("Obs", "Pred"))

# svm
model.svm <- svm(email_class ~ ., data = train.bin.DF)
pred.svm <- predict(model.svm, test.bin.DF, type = "email_class")
table.svm <- table(test.bin.DF$email_class, pred.svm, dnn=c("Obs", "Pred"))

results_svm <- svm_experiment(2, data.tfidf.selected_features[1:1000,], 'linear', NA, 1, 10)
results_svm <- svm_rbf_gamma_tests(2, data.tfidf.selected_features[1:1000,], c(0.05))
results_svm <- svm_polynomial_degree_tests(2, data.tfidf.selected_features[1:1000,], c(2, 3))
#################################

# eksperymenty związane z różną reprezentacją danych
# tree
bin.model.tree <- rpart(email_class~., method="email_class", data = train.bin.DF)
bin.pred.tree <- predict(bin.model.tree, test.bin.DF, type="email_class")
table(test.bin.DF$email_class, bin.pred.tree, dnn=c("Obs", "Pred"))
prp(bin.model.tree)

# ponizej tabela dla TREE (wszystkie argumenty, TF-IDF, )
      #Pred
#Obs     ham spam
#ham  1338   62
#spam  599  798

# ponizej tabela dla TREE (wszystkie argumenty, BIN, )
      #Pred
#Obs     ham spam
#ham  1348   52
#spam  538  859

# wniosek: zmiana reprezentacji na binarna polepszyło nieznacznie model predykcyjny
# drzewa, widac ze mniej maili 'ham' zostalo rozpoznanych jako 'spam'
