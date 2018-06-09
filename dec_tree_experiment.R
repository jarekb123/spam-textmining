if (! "dismo" %in% row.names(installed.packages()))
  install.packages("dismo")
library(dismo)
if (! "parallel" %in% row.names(installed.packages()))
  install.packages("parallel")
library(parallel)


decision_tree_experiment <- function(k, emails, words, max_depth, min_split, features_number=200) {
  # emails have to be sorted in decreasing order
  #words <- rownames(emails)[1:features_number]
  fmla <- as.formula(paste("class ~ ", paste(words, collapse = "+")))
  #print(fmla)
  #########
  folds <- kfold(emails, k)
  
  result = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(result) <- c('max_depth', 'min_split', 'true_ham', 'false_ham', 'true_spam', 'false_spam', 'classification_error')
  
  for(i in 1:k){
    test_fold <- i
    
    testData <- emails[folds == test_fold, ]
    trainData <- emails[folds != test_fold, ]
    
    model.tree <- rpart(fmla, data=trainData, minsplit = min_split, maxdepth = max_depth)
    prp(model.tree)
    pred.tree <- predict(model.tree, testData, type="class")
    result_table <- table(testData$class, pred.tree, dnn=c("Obs", "Pred"))
    classification_error <- sum(pred.tree != testData$class) / NROW(pred.tree)
    
    result[nrow(result) + 1,] <- list(max_depth, min_split, result_table[1, 1],
                                      result_table[2, 1], result_table[2, 2],
                                      result_table[1, 2], classification_error)
  }
  result <- colMeans(result)
  return(result)
}

decision_tree_grid_search_tests <- function(k, emails, words, max_depths, 
                                                      min_splits, features_number=200) {
  results = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(results) <- c('max_depth', 'min_split', 'true_ham', 'false_ham',
                         'true_spam', 'false_spam', 'classification_error')
  
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  args <- expand.grid(max_depths, min_splits)
  #args.list <- split(args, seq(nrow(args)))
  results <- mcmapply(function(max_depth, min_split) {return(decision_tree_experiment(k, emails, words, 
                                                                                      max_depth, min_split,
                                                                                      features_number))},
                      as.list(args[, 1]), as.list(args[, 2]), mc.cores = cores_number)
  results <- data.frame(t(results))
  #plot(results[,1], xlab = 'Max depth', result[, 4], ylab = 'Error')
  return(results)
}

decision_tree_max_depths_tests <- function(k, emails, words, max_depths, 
                                           min_split, features_number=200) {
  results = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(results) <- c('max_depth', 'min_split', 'true_ham', 'false_ham',
                         'true_spam', 'false_spam', 'classification_error')

  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  results <- parLapply(cluster, max_depths, function(max_depth) {return(decision_tree_experiment(k, emails, words, 
                                                                          max_depth, min_split,
                                                                          features_number))})
  results <- do.call("rbind", results)
  plot(results[,1], xlab = 'Max depth', results[, 7], ylab = 'Error')
  return(results)
}

decision_tree_min_splits_tests <- function(k, emails, words, max_depth, 
                                           min_splits, features_number=200) {
  results = data.frame(matrix(ncol = 7, nrow = 0))
  colnames(results) <- c('max_depth', 'min_split', 'true_ham', 'false_ham',
                         'true_spam', 'false_spam', 'classification_error')
  
  cores_number <- detectCores() - 1
  cluster <- makeCluster(cores_number, type="FORK")
  #clusterExport(cl=cl, varlist=c("features_number"))
  results <- parLapply(cluster, min_splits, function(min_split) {return(decision_tree_experiment(k, emails, words, 
                                                                                                 max_depth, min_split,
                                                                                                 features_number))})
  results <- do.call("rbind", results)
  plot(results[,2], xlab = 'Min split', results[, 7], ylab = 'Error')
  return(results)
}