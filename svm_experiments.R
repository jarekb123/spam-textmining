if (! "e1071" %in% row.names(installed.packages()))
  install.packages("e1071")
library(e1071)

# one of rbf_sigma, polynomial_degree will be used, depending on the kernel
svm_experiment <- function(k, emails, kernel, rbf_sigma, polynomial_degree,
                           cost) {
  folds <- kfold(emails, k)
  
  result = data.frame(matrix(ncol = 8, nrow = 0))
  colnames(result) <- c('rbf_sigma', 'polynomial_degree', 'cost',
                        'true_ham', 'false_ham', 'true_spam', 'false_spam',
                        'classification_error')
  for(i in 1:k){
    test_fold <- i
    
    testData <- emails[folds == test_fold, ]
    trainData <- emails[folds != test_fold, ]
    
    if (kernel == 'linear') {
      model.svm <- svm(email_class ~ ., data = trainData, kernel = 'linear')
    } else if (kernel == 'radial basis') {
      model.svm <- svm(email_class ~ ., data = trainData, kernel = 'radial basis', gamma = gamma,
                       cost = cost)
    } else if (kernel == 'polynomial'){
      model.svm <- svm(email_class ~ ., data = trainData, kernel = 'polynomial' ,
                       degree = polynomial_degree, cost = cost)
    }
    pred.svm <- predict(model.svm, testData)
    result_table <- table(testData$email_class, pred.svm, dnn=c("Obs", "Pred"))
    
    classification_error <- sum(pred.svm != testData$email_class) / NROW(pred.svm)
    
    result[nrow(result) + 1,] <- list(rbf_sigma, polynomial_degree, cost,
                                      result_table[1, 1], result_table[2, 1],
                                      result_table[2, 2], result_table[1, 2],
                                      classification_error)
  }
  result <- colMeans(result)
  result['kernel'] <- kernel  
  return(result)
}
