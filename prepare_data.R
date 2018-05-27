require("tm")
require("tm.plugin.mail")
require(tidytext)

prepare_corpus <- function(corpus) {
  corpus <- tm_map(corpus, removeMultipart)
  corpus <- tm_map(corpus, content_transformer(function(x) iconv(x, to = "UTF-8", sub = "byte")))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, removePunctuation)
  return(corpus)
}



non_spam <-  Corpus(DirSource("easy_ham", encoding = "UTF-8"), readerControl = list(reader = readMail, language="en"))
non_spam <- prepare_corpus(non_spam)
non_spam.TF <- DocumentTermMatrix(non_spam, control = list(weighting = weightTf, minDocFreq=3))
non_spam.TFIDF <- DocumentTermMatrix(non_spam, control = list(weigthing = weightTfIdf, minDocFreq=3))

spam <- Corpus(DirSource("spam"), readerControl = list(reader = readMail, language="en"))
spam <- prepare_corpus(spam)
spam.TF <- DocumentTermMatrix(spam, control = list(weigthing = weightTf, minDocFreq=3))
spam.TFIDF <- DocumentTermMatrix(spam, control = list(weigthing = weightTfIdf, minDocFreq=3))

# converting to DT Data Frame:
non_spam.TFIDF.df <- tidy(non_spam.TFIDF)
spam.TFIDF.df <- tidy(spam.TFIDF)