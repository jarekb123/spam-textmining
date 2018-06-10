if (! "dismo" %in% row.names(installed.packages()))
  install.packages("dismo")
library(dismo)
if (! "parallel" %in% row.names(installed.packages()))
  install.packages("parallel")
library(parallel)
if (! "rpart" %in% row.names(installed.packages()))
  install.packages("rpart")
library(rpart)


decision_tree_experiment <- function(k, emails, max_depth, min_split) {
  words <- colnames(emails)
  fmla <- as.formula(paste("email_class ~ ", paste(words, collapse = "+")))
  #########
  folds <- kfold(emails, k)
  
  result = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(result) <- c('max_depth', 'min_split', 'true_ham', 'false_ham', 'true_spam', 'false_spam', 'classification_error')
  
  for(i in 1:k){
    test_fold <- i
    
    testData <- emails[folds == test_fold, ]
    trainData <- emails[folds != test_fold, ]
    
    model.tree <- rpart(fmla, data = trainData, minsplit = min_split, maxdepth = max_depth)
    #prp(model.tree)
    pred.tree <- predict(model.tree, testData, type="class")
    result_table <- table(testData$email_class, pred.tree, dnn=c("Obs", "Pred"))
    classification_error <- sum(pred.tree != testData$email_class) / NROW(pred.tree)
    
    result[nrow(result) + 1,] <- list(max_depth, min_split, result_table[1, 1],
                                      result_table[2, 1], result_table[2, 2],
                                      result_table[1, 2], classification_error)
  }
  result <- colMeans(result)
  return(result)
}

decision_tree_grid_search_tests <- function(k, emails, max_depths, 
                                            min_splits) {
  results = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(results) <- c('max_depth', 'min_split', 'true_ham', 'false_ham',
                         'true_spam', 'false_spam', 'classification_error')
  
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  grid <- expand.grid(max_depths, min_splits)
  grid.list <- split(grid, seq(nrow(grid)))

  results <- mcmapply(function(max_depth, min_split) {return(decision_tree_experiment(k, emails,
                                                                                      max_depth, min_split
                                                                                      ))},
                      as.list(grid[, 1]), as.list(grid[, 2]), mc.cores = cores_number)
  results <- data.frame(t(results))
  #plot(results[,1], xlab = 'Max depth', result[, 4], ylab = 'Error')
  return(results)
}

decision_tree_max_depths_tests <- function(k, emails, max_depths, 
                                           min_split) {
  results = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(results) <- c('max_depth', 'min_split', 'true_ham', 'false_ham',
                         'true_spam', 'false_spam', 'classification_error')

  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  results <- parLapply(cluster, max_depths, function(max_depth) {return(decision_tree_experiment(k, emails,
                                                                          max_depth, min_split
                                                                          ))})
  results <- do.call("rbind", results)
  plot(results[,1], xlab = 'Max depth', results[, 7], ylab = 'Error')
  return(results)
}

decision_tree_min_splits_tests <- function(k, emails, max_depth, 
                                           min_splits) {
  results = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(results) <- c('max_depth', 'min_split', 'true_ham', 'false_ham',
                         'true_spam', 'false_spam', 'classification_error')
  
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  #clusterExport(cl=cl, varlist=c("features_number"))
  results <- parLapply(cluster, min_splits, function(min_split) {return(decision_tree_experiment(k, emails,
                                                                                                 max_depth, min_split))})
  results <- do.call("rbind", results)
  plot(results[,2], xlab = 'Min split', results[, 7], ylab = 'Error')
  return(results)
}