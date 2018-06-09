if (! "dismo" %in% row.names(installed.packages()))
  install.packages("dismo")
library(dismo)


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