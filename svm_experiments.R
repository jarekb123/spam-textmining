if (! "e1071" %in% row.names(installed.packages()))
  install.packages("e1071")
library(e1071)
if (! "parallel" %in% row.names(installed.packages()))
  install.packages("parallel")
library(parallel)
if (! "dismo" %in% row.names(installed.packages()))
  install.packages("dismo")
library(dismo)

# one of rbf_gamma, polynomial_degree will be used, depending on the kernel
svm_experiment <- function(k, emails, kernel, rbf_gamma, polynomial_degree,
                           cost) {
  folds <- kfold(emails, k)
  
  result = data.frame(matrix(ncol = 8, nrow = 0))
  colnames(result) <- c('rbf_gamma', 'polynomial_degree', 'cost',
                        'true_ham', 'false_ham', 'true_spam', 'false_spam',
                        'classification_error')
  for(i in 1:k){
    test_fold <- i
    
    testData <- emails[folds == test_fold, ]
    trainData <- emails[folds != test_fold, ]
    
    # cost = 1 by default (e1071 doc)
    if (kernel == 'linear') {
      model.svm <- svm(email_class ~ ., data = trainData, kernel = 'linear', cost = cost)
    } else if (kernel == 'radial') {
      model.svm <- svm(email_class ~ ., data = trainData, kernel = 'radial', gamma = rbf_gamma,
                       cost = cost)
    } else if (kernel == 'polynomial'){
      model.svm <- svm(email_class ~ ., data = trainData, kernel = 'polynomial' ,
                       degree = polynomial_degree, cost = cost)
    }
    pred.svm <- predict(model.svm, testData)
    result_table <- table(testData$email_class, pred.svm, dnn=c("Obs", "Pred"))
    
    classification_error <- sum(pred.svm != testData$email_class) / NROW(pred.svm)
    
    result[nrow(result) + 1,] <- list(rbf_gamma, polynomial_degree, cost,
                                      result_table[1, 1], result_table[2, 1],
                                      result_table[2, 2], result_table[1, 2],
                                      classification_error)
  }
  result <- colMeans(result)
  result['kernel'] <- kernel  
  return(result)
}


svm_rbf_gamma_tests <- function(k, emails, gammas) {
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  results <- parLapply(cluster, gammas, function(rbf_gamma) 
    {return(svm_experiment(k, emails, "radial", rbf_gamma, 1,
                           1))})
  results <- do.call("rbind", results)
  plot(results[,1], xlab = 'RBF gamma', results[, 8], ylab = 'Error')
  return(results)
}

svm_polynomial_degree_tests <- function(k, emails, degrees) {
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  results <- parLapply(cluster, degrees, function(degree) 
  {return(svm_experiment(k, emails, "polynomial", 0, degree,
                         1))})
  results <- do.call("rbind", results)
  plot(results[,2], xlab = 'Polynomial degree', results[, 8], ylab = 'Error')
  return(results)
}