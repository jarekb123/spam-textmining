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


spam.dir <- "~/studia/mgr/mow/project/spamassasin/"

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

easy_ham.df$class <- factor("ham")
easy_ham_2.df$class <- factor("ham")
hard_ham.df$class <- factor("ham")
spam.df$class <- factor("spam")
spam_2.df$class <- factor("spam")

names(easy_ham.df) <- c("text", "class")
names(easy_ham_2.df) <- c("text", "class")
names(hard_ham.df) <- c("text", "class")
names(spam_2.df) <- c("text", "class")
names(spam.df) <- c("text", "class")

train.data <- rbind(easy_ham.df, hard_ham.df, spam.df)
train.num <- nrow(train.data)
train.test <- rbind(train.data, easy_ham_2.df, spam_2.df)
names(train.data) <- c("text", "class")



set.seed(1000)
train_out.data <- train.data$class
train_text.data <- train.data$text

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

for(i in 1: nrow(train.data)) {
  train.test$text[i] <- removeHTMLClosingTag(removeHTMLAttr(train.test$text[i]))
  train.test$isContainingHTML[i] <- isContainingHTMLTags(train.test$text[i])
}

#dtm.tfidf.test <- create_matrix(train.test$text, language = "english", minWordLength = 3, removeNumbers = TRUE, stemWords = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.95, weighting = weightTfIdf)
##############################################
# dane do modeli predykcyjnych - TFIDF
dtm.tfidf <- create_matrix(train.test$text, 
                           language = "english", 
                           minWordLength = 3, 
                           removeNumbers = TRUE, 
                           stemWords = TRUE, 
                           removePunctuation = TRUE,
                           removeSparseTerms = 0.99, 
                           weighting = weightTfIdf
)
data.tfidf <- cbind(train.test, as.matrix(dtm.tfidf))
names(data.tfidf) <- make.names(names(data.tfidf)) 

train.tfidf.DF <- data.tfidf[1:train.num,]
test.tfidf.DF <- data.tfidf[-(1:train.num),]

train.tfidf.DF$text = NULL
test.tfidf.DF$text = NULL
###########################################
# dane do modeli predykcyjnych - TF
dtm.tf <- create_matrix(train.test$text, 
                        language = "english", 
                        minWordLength = 3, 
                        removeNumbers = TRUE, 
                        stemWords = TRUE, 
                        removePunctuation = TRUE,
                        removeSparseTerms = 0.99, 
                        weighting = weightTf
)
data.tf <- cbind(train.test, as.matrix(dtm.tf))
names(data.tf) <- make.names(names(data.tf)) 

train.tf.DF <- data.tf[1:train.num,]
test.tf.DF <- data.tf[-(1:train.num),]

train.tf.DF$text = NULL
test.tf.DF$text = NULL
########################################
# binarna reprezentacja - 0 lub 1
dtm.bin <- create_matrix(train.test$text, 
                        language = "english", 
                        minWordLength = 3, 
                        removeNumbers = TRUE, 
                        stemWords = TRUE, 
                        removePunctuation = TRUE,
                        removeSparseTerms = 0.99, 
                        weighting = weightBin
)
data.bin <- cbind(train.test, as.matrix(dtm.bin))
names(data.bin) <- make.names(names(data.bin)) 

train.bin.DF <- data.bin[1:train.num,]
test.bin.DF <- data.bin[-(1:train.num),]

train.bin.DF$text = NULL
test.bin.DF$text = NULL



############################################
# drzewo decyzyjne
model.tree <- rpart(class~., method="class", data = train.tfidf.DF)
pred.tree <- predict(model.tree, test.tfidf.DF, type = "class")
table(test.tfidf.DF$class, pred.tree, dnn=c("Obs", "Pred"))
prp(model.tree)

# selekcja atrybutów
# variable importance z drzewa decyzyjnego
varImportance.tree <- varImp(model.tree)
# varImp z randomForest
# ref: https://www.r-bloggers.com/variable-importance-plot-and-variable-selection/
model.rf <- randomForest(class~., data = train.tfidf.DF)
varImportance.rf <- varImp(model.rf)

varImportance.tree.sorted <- data.frame(varImportance.tree, rownames(varImportance.tree))
varImportance.tree.sorted <- varImportance.tree.sorted[order(-(varImportance.tree.sorted$Overall)),]

varImportance.rf.sorted <- data.frame(varImportance.rf, rownames(varImportance.rf))
varImportance.rf.sorted <- varImportance.rf.sorted[order(-(varImportance.rf.sorted$Overall)),]


# model po selekcji 25 atrybutów z randomForest
words <- rownames(varImportance.rf.sorted)[1:25]
fmla <- as.formula(paste("class ~ ", paste(words, collapse = "+")))

model.tree.25 <- rpart(fmla, data=train.tfidf.DF)
prp(model.tree.25)
pred.tree.25 <- predict(model.tree.25, test.tfidf.DF, type="class")
table(test.tfidf.DF$class, pred.tree.25, dnn=c("Obs", "Pred"))


model.tree.cp <- rpart(class~., data=train.tfidf.DF, cp = 0.02, minbucket = 30)
pred.tree.cp <- predict(model.tree.cp, test.tfidf.DF, type = "class")
table(test.tfidf.DF$class, pred.tree.cp, dnn=c("Obs", "Pred"))
prp(model.tree.cp)

# klasyfikator Bayesa
model.bayes <- naiveBayes(class~., data = train.tfidf.DF)
pred.bayes <- predict(model.bayes, test.tfidf.DF, type = "class")
table(pred = pred.bayes, true = test.tfidf.DF$class, dnn=c("Obs", "Pred"))

# svm
model.svm <- svm(class~., data = train.tfidf.DF)
pred.svm <- predict(model.svm, test.tfidf.DF)
table.svm <- table(pred = pred.svm, ALabels = test.tfidf.DF$class, dnn=c("Obs", "Pred"))
#################################

# eksperymenty związane z różną reprezentacją danych
# tree
bin.model.tree <- rpart(class~., method="class", data = train.bin.DF)
bin.pred.tree <- predict(bin.model.tree, test.bin.DF, type="class")
table(test.bin.DF$class, bin.pred.tree, dnn=c("Obs", "Pred"))
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
