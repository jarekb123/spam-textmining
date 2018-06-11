if (! "e1071" %in% row.names(installed.packages()))
  install.packages("e1071")
library(e1071)
if (! "parallel" %in% row.names(installed.packages()))
  install.packages("parallel")
library(parallel)
if (! "dismo" %in% row.names(installed.packages()))
  install.packages("dismo")
library(dismo)

bayes_cost_sensitive_pred <- function(bayes_model, emails, ham_as_spam_cost, spam_as_ham_cost) {
  pred.bayes <- predict(bayes_model, emails, type = "raw")
  
  # P(ham|x) > ham_threshold_coef * P(spam|x) -> klasyfikuj jako ham
  ham_threshold_coef <- spam_as_ham_cost / ham_as_spam_cost
  result.pred  <- ifelse (pred.bayes[, 1] > pred.bayes[, 2] * ham_threshold_coef, "ham", "spam")
  
  return (result.pred)
}

avg_cost <- function(pred_classes, classes, ham_as_spam_cost, spam_as_ham_cost) {
  confusion_matrix <- table(pred = pred_classes, true = classes, dnn=c("Obs", "Pred"))
  cost <- ham_as_spam_cost * confusion_matrix[1, 2]
  cost <- cost + spam_as_ham_cost * confusion_matrix[2, 1]
  return(cost / length(pred_classes))
}

naive_bayes_experiment <- function(k, emails, ham_as_spam_cost = 1, spam_as_ham_cost = 1) {
  folds <- kfold(emails, k)
  
  result = data.frame(matrix(ncol = 8, nrow = 0))
  colnames(result) <- c('spam_as_ham_cost', 'ham_as_spam_cost', 'avg_cost', 'true_ham', 
                        'false_ham', 'true_spam', 'false_spam', 'classification_error')
  
  for (i in 1:k) {
    test_fold <- i

    testData <- emails[folds == test_fold, ]
    trainData <- emails[folds != test_fold, ]

    model.bayes <- naiveBayes(email_class~., data = trainData, laplace = 1)
    pred.bayes <- bayes_cost_sensitive_pred(model.bayes, testData, ham_as_spam_cost, spam_as_ham_cost)

    result_table <- table(true = testData$email_class, pred = pred.bayes, dnn=c("Obs", "Pred"))
    classification_error <- sum(pred.bayes != testData$email_class) / NROW(pred.bayes)
    pred_avg_cost <- avg_cost(pred.bayes, testData$email_class, ham_as_spam_cost, spam_as_ham_cost)

    result[nrow(result) + 1,] <- list(spam_as_ham_cost, ham_as_spam_cost, pred_avg_cost,
                                      result_table[1, 1], result_table[2, 1],
                                      result_table[2, 2], result_table[1, 2],
                                      classification_error)
  }
  result <- colMeans(result)
  return(result)
}

naive_bayes_misclassification_costs_tests <- function(k, emails,
                                                      spam_as_ham_cost_to_ham_as_spam_ratio) {
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  results <- parLapply(cluster, spam_as_ham_cost_to_ham_as_spam_ratio, function(ratio) 
    {return(naive_bayes_experiment(k, emails, ham_as_spam_cost = 1,
                                   spam_as_ham_cost = ratio))})
  results <- do.call("rbind", results)
  plot(results[,1], xlab = 'Spam as ham to ham as spam cost ratio', results[, 8],
       ylab = 'Error')
  return(results)
}