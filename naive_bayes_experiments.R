if (! "e1071" %in% row.names(installed.packages()))
  install.packages("e1071")
library(e1071)

bayes_cost_sensitive <- function(bayes_model, emails, ham_as_spam_cost, spam_as_ham_cost) {
  pred.bayes <- predict(bayes_model, test.tfidf.DF.selected_features, type = "raw")
  
  # P(ham|x) > ham_threshold_coef * P(spam|x) -> classify as ham
  ham_threshold_coef <- spam_as_ham_cost / ham_as_spam_cost
  result.pred  <- ifelse (pred.bayes[, 1] > pred.bayes[, 2] * ham_threshold_coef, "ham", "spam")
  
  return (pred.classes)
}

avg_cost <- function(pred_classes, classes, ham_as_spam_cost, spam_as_ham_cost) {
  confusion_matrix <- table(pred = pred_classes, true = classes, dnn=c("Obs", "Pred"))
  cost <- ham_as_spam_cost * confusion_matrix[1, 2]
  cost <- cost + spam_as_ham_cost * confusion_matrix[2, 1]
  return(cost / length(pred_classes))
}
