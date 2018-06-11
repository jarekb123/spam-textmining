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

# przed uruchomieniem projektu należy wypakować pobrany zbiór danych do nowego folderu spamassasin
spam.dir <- "./spamassasin/"

get.msg <- function(path.dir) {
  con <- file(path.dir, open="rt", encoding = "latin1")
  text <- readLines(con)
  msg <- text[seq(which(text=="")[1]+1, length(text), 1)]
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

set.seed(12345)

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
data.tfidf$text <- NULL

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


varImportance.tf <- attrEval(email_class~., data.tf, estimator="Gini")
varImportance.tf <- data.frame(varImportance.tf, names(varImportance.tf))
names(varImportance.tf) <- c("importance", "term")
varImportance.tf.decreasing <- varImportance.tf[order(varImportance.tf$importance, decreasing = TRUE),]
varImportance.tf.decreasing.columns <- order(varImportance.tf$importance, decreasing = TRUE)[1:features_num]
data.tf.selected_features <- data.tf[, varImportance.tf.decreasing.columns[1:features_num] + 1]
data.tf.selected_features$email_class <- data.tf$email_class


varImportance.bin <- attrEval(email_class~., data.bin, estimator="Gini")
varImportance.bin <- data.frame(varImportance.bin, names(varImportance.bin))
names(varImportance.bin) <- c("importance", "term")
varImportance.bin.decreasing <- varImportance.bin[order(varImportance.bin$importance, decreasing = TRUE),]
varImportance.bin.decreasing.columns <- order(varImportance.bin$importance, decreasing = TRUE)[1:features_num]
data.bin.selected_features <- data.bin[, varImportance.bin.decreasing.columns[1:features_num] + 1]
data.bin.selected_features$email_class <- data.bin$email_class
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

bayes_y_range <- range(bayes_results_bin[, 8] * 100, 
                       bayes_results_tf[, 8] * 100,
                       bayes_results_tf_idf[, 8] * 100)
plot(bayes_results_tf_idf[, 1], bayes_results_tf_idf[, 8] * 100, log='x', col='red', 
     ylim=bayes_y_range, ylab = 'Błąd klasyfikacji [%]',
     xlab=expression(frac("Koszt błędnej klasyfikacji spamu", "Koszt błędnej klasyfikacji pożądanej korespondencji")),
     mgp = c(3, 0.1, 0))
points(bayes_results_tf[, 1], bayes_results_tf[, 8] * 100, col='green')
points(bayes_results_bin[, 1], bayes_results_bin[, 8] * 100, col='black')
legend('center', legend=c("tf-idf", "tf", "binarna"),
      col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
      title="Reprezentacja tekstu")

##############################
#TESTY DRZEW DECYZYJNYCH
tree_results_tf_idf <- decision_tree_grid_search_tests(5, data.tfidf.selected_features, 
                                                       c(1, 3, 5, 10, 15, 20, 25, 30), 
                                                       c(1, 5, 10, 25, 50))
tree_results_tf <- decision_tree_grid_search_tests(5, data.tf.selected_features, 
                                                       c(1, 3, 5, 10, 15, 20, 25, 30), 
                                                       c(1, 5, 10, 25, 50))
tree_results_bin <- decision_tree_grid_search_tests(5, data.bin.selected_features, 
                                                       c(1, 3, 5, 10, 15, 20, 25, 30), 
                                                       c(1, 5, 10, 25, 50))
tree_results_tf_idf_overfitting_test <- decision_tree_grid_search_tests(5, data.tfidf.selected_features, 
                                                       c(30), 
                                                       c(1))

plot(tree_results_tf_idf[tree_results_tf_idf[,2] == 5, 1],
     tree_results_tf_idf[tree_results_tf_idf[,2] == 5, 7] * 100,
      col='red', 
     ylim=range(7.5, 11.5),
     xlim=range(3, 30),
     ylab = 'Błąd klasyfikacji [%]',
     xlab="Maksymalna głębokość drzewa",
     mgp = c(3, 0.1, 0))
points(tree_results_tf[tree_results_tf[,2] == 5, 1], 
       tree_results_tf[tree_results_tf[,2] == 5, 7] * 100, col='green')
points(tree_results_bin[tree_results_bin[,2] == 5, 1],
       tree_results_bin[tree_results_bin[,2] == 5, 7] * 100, col='black')
legend('bottomleft', legend=c("tf-idf", "tf", "binarna"),
       col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
       title="Reprezentacja tekstu")

plot(tree_results_tf_idf[tree_results_tf_idf[,1] == 10, 2],
     tree_results_tf_idf[tree_results_tf_idf[,1] == 10, 7] * 100,
     log='x', col='red', 
     ylim=range(7.5, 9),
     xlim=range(1.0, 51.0),
     ylab = 'Błąd klasyfikacji [%]',
     xlab="Minimalny rozmiar podziału",
     mgp = c(3, 0.1, 0))
points(tree_results_tf[tree_results_tf[,1] == 10, 2], 
       tree_results_tf[tree_results_tf[,1] == 10, 7] * 100, col='green')
points(tree_results_bin[tree_results_bin[,1] == 10, 2],
       tree_results_bin[tree_results_bin[,1] == 10, 7] * 100, col='black')
legend('bottomleft', legend=c("tf-idf", "tf", "binarna"),
       col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
       title="Reprezentacja tekstu")

############### SVM ###############
results_svm_tf_idf_gammas <- svm_rbf_gamma_tests(5, data.tfidf.selected_features,
                                                 c(0.001, 0.005, 0.01, 0.025, 0.05))

results_svm_tf_gammas <- svm_rbf_gamma_tests(5, data.tf.selected_features,
                                                 c(0.001, 0.005, 0.01, 0.025, 0.05))

results_svm_bin_gammas <- svm_rbf_gamma_tests(5, data.bin.selected_features,
                                                 c(0.001, 0.005, 0.01, 0.025, 0.05))

plot(results_svm_tf_idf_gammas[, 1],
     as.numeric(results_svm_tf_idf_gammas[, 8]) * 100,
     log='x', col='red', 
     ylim=range(1, 11),
     xlim=range(0.001, 0.05),
     ylab = 'Błąd klasyfikacji [%]',
     xlab="Paramter gamma jądra w postaci radialnej funkcji bazowej",
     mgp = c(3, 0.1, 0))
points(results_svm_tf_gammas[, 1], 
       as.numeric(results_svm_tf_gammas[, 8]) * 100, col='green')
points(results_svm_bin_gammas[, 1],
       as.numeric(results_svm_bin_gammas[, 8]) * 100, col='black')
legend('bottomleft', legend=c("tf-idf", "tf", "binarna"),
       col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
       title="Reprezentacja tekstu")


results_svm_tf_idf_polynomial <- svm_polynomial_degree_tests(5, data.tfidf.selected_features,
                                           c(1, 2, 3))

results_svm_tf_polynomial <- svm_polynomial_degree_tests(5, data.tf.selected_features,
                                                             c(1, 2, 3))

results_svm_bin_polynomial <- svm_polynomial_degree_tests(5, data.bin.selected_features,
                                                             c(1, 2, 3))

plot(results_svm_tf_idf_polynomial[, 2],
     as.numeric(results_svm_tf_idf_polynomial[, 8]) * 100,
     col='red', 
     ylim=range(1, 19),
     xlim=range(1, 3),
     ylab = 'Błąd klasyfikacji [%]',
     xlab="Stopień wielomianu jądra wielomianowego",
     mgp = c(3, 0.1, 0))
points(results_svm_tf_polynomial[, 2], 
       as.numeric(results_svm_tf_polynomial[, 8]) * 100, col='green')
points(results_svm_bin_polynomial[, 2],
       as.numeric(results_svm_bin_polynomial[, 8]) * 100, col='black')
legend('bottomleft', legend=c("tf-idf", "tf", "binarna"),
       col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
       title="Reprezentacja tekstu")

results_svm_tf_idf_rbf_cost <- svm_cost_rbf_tests(5, data.tfidf.selected_features, 0.005,
                                              c(0.1, 1, 10, 100))

results_svm_tf_rbf_cost <- svm_cost_rbf_tests(5, data.tf.selected_features, 0.005,
                                                  c(0.1, 1, 10, 100))

results_svm_bin_rbf_cost <- svm_cost_rbf_tests(5, data.bin.selected_features, 0.005,
                                                  c(0.1, 1, 10, 100))
plot(results_svm_tf_idf_rbf_cost[, 3],
     as.numeric(results_svm_tf_idf_rbf_cost[, 8]) * 100,
     log='x', col='red', 
     ylim=range(1, 11),
     xlim=range(0.1, 100),
     ylab = 'Błąd klasyfikacji [%]',
     xlab="Koszt naruszenia marginesu funkcji decyzyjnej",
     mgp = c(3, 0.1, 0))
points(results_svm_tf_rbf_cost[, 3], 
       as.numeric(results_svm_tf_rbf_cost[, 8]) * 100, col='green')
points(results_svm_bin_rbf_cost[, 3],
       as.numeric(results_svm_bin_rbf_cost[, 8]) * 100, col='black')
legend('bottomleft', legend=c("tf-idf", "tf", "binarna"),
       col=c("red", "green", "black"), pch=c(1, 1, 1), xjust = 0,
       title="Reprezentacja tekstu")