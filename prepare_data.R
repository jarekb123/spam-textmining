if (! "tm" %in% row.names(installed.packages()))
  install.packages("tm")
library("tm")
if (! "tm.plugin.mail" %in% row.names(installed.packages()))
  install.packages("tm.plugin.mail")
library("tm.plugin.mail")
if (! "tidytext" %in% row.names(installed.packages()))
  install.packages("tidytext")
library("tidytext")


prepare_corpus <- function(corpus) {
  corpus <- tm_map(corpus, removeMultipart)
  #necessary?
  corpus <- tm_map(corpus, content_transformer(function(x) iconv(x, to = "UTF-8", sub = "byte")))
  #apart from stripWhitespace, all the operations below may be replaced with specifying proper options in control list passed to DocumentTermMatrix()
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, removePunctuation)
  return(corpus)
}

non_spam <-  Corpus(DirSource("easy_ham", encoding = "UTF-8"), readerControl = list(reader = readMail, language="en"))
non_spam <- prepare_corpus(non_spam)
non_spam.TF <- DocumentTermMatrix(non_spam, control = list(weighting = weightTf))
non_spam.TFIDF <- DocumentTermMatrix(non_spam, control = list(weighting = weightTfIdf))
non_spam.BIN <- DocumentTermMatrix(non_spam, control = list(weighting = weightBin))

spam <- Corpus(DirSource("spam"), readerControl = list(reader = readMail, language="en"))
#spam[[38]] causes the error - it contains invalid utf8 characters

spam <- prepare_corpus(spam)
spam.TF <- DocumentTermMatrix(spam, control = list(weighting = weightTf))
spam.TFIDF <- DocumentTermMatrix(spam, control = list(weighting = weightTfIdf))
spam.BIN <- DocumentTermMatrix(spam, control = list(weighting = weightTf))
# converting to DT Data Frame:
non_spam.TFIDF.df <- tidy(non_spam.TFIDF)
spam.TFIDF.df <- tidy(spam.TFIDF)