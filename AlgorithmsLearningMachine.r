library(RWeka)
library(foreign)
library(e1071)
library(randomForest)
library(caret)

J48Pruned <- function(sourceFile){
  
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)

  result <-J48(arffData$Classifier~., data= arffData, control= c("-C", "0.25", "-M", "2"))
  evaluation <- evaluate_Weka_classifier(result,cost = NULL, numFolds = 10, complexity = TRUE, seed = 1, class = TRUE)
  print(evaluation)
 
}

J48Unpruned <- function(sourceFile){
  
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  
  result <-J48(arffData$Classifier~., data= arffData, control= c("-U", "-M", "2"))
  evaluation <- evaluate_Weka_classifier(result,cost = NULL, numFolds = 10, complexity = TRUE, seed = 1, class = TRUE)
  print(evaluation)
  
}

J48ReduceError <- function(sourceFile){
  
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  
  result <-J48(arffData$Classifier~., data= arffData, control= c("-R", "-N", "3", "-Q", "1", "-M", "2"))
  evaluation <- evaluate_Weka_classifier(result,cost = NULL, numFolds = 10, complexity = TRUE, seed = 1, class = TRUE)
  print(evaluation)
}

NaiveBayes <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    nb = naiveBayes(trainData$Classifier~., data = trainData)
    prediction = predict(nb,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  
  print(matrix_output)
}

RandomForest <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)

  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    forest =  randomForest::randomForest(trainData$Classifier~., data = trainData, importance=TRUE, ntree=100)
    prediction = predict(forest,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }

  print(matrix_output)
}

Jrip <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    jrip <- train(trainData, trainData$Classifier, method = "JRip")
    prediction <- predict(jrip,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

cSVMRadial <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="radial", scale = FALSE, type = "C-classification")
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

cSVMPoly <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="polynomial", scale = FALSE,type = "C-classification")
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

cSVMLinear <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="linear", scale = FALSE,type = "C-classification")
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

cSVMSigmoid <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="sigmoid", scale = FALSE,type = "C-classification")
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

vSVMRadial <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="radial", scale = FALSE, type = "nu-classification", nu = 0.5)
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

vSVMPoly <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="polynomial", scale = FALSE,type = "nu-classification", nu = 0.5)
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

vSVMLinear <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="linear", scale = FALSE,type = "nu-classification", nu = 0.5)
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

vSVMSigmoid <- function(sourceFile){
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  set.seed(1)
  arffData<-arffData[sample(nrow(arffData)),]
  folds <- cut(seq(1,nrow(arffData)),breaks=10,labels=FALSE)
  matrix_output <- matrix(c(0,0,0,0),2)
  
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- arffData[testIndexes, ]
    trainData <- arffData[-testIndexes, ]
    #Use the test and train data partitions however you desire...
    svm_model_result <- svm(trainData$Classifier~., data=trainData, kernel="sigmoid", scale = FALSE,type = "nu-classification", nu = 0.5)
    prediction <- predict(svm_model_result,testData)
    matrix_output <- matrix_output+as.data.frame.matrix(table(prediction,testData$Classifier))
  }
  print(matrix_output)
}

SMOPolyKernel <- function(sourceFile){
  
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  result <-SMO(arffData$Classifier~., data= arffData,   Weka_control(K = list("weka.classifiers.functions.supportVector.RBFKernel", G = 2)))
  evaluation <- evaluate_Weka_classifier(result,cost = NULL, numFolds = 10, complexity = TRUE, seed = 1, class = TRUE)
  print(evaluation)
  
}

SMORBFKernel <- function(sourceFile){
  
  help.search(sourceFile)
  arffData <- read.arff(file=sourceFile)
  
  result <-SMO(arffData$Classifier~., data= arffData,   Weka_control(K = list("weka.classifiers.functions.supportVector.PolyKernel", G = 2)))
  evaluation <- evaluate_Weka_classifier(result,cost = NULL, numFolds = 10, complexity = TRUE, seed = 1, class = TRUE)
  print(evaluation)

}

#dataSet <- c("/home/joaolucas/AuthAlgorithms.arff","/home/joaolucas/Bypass.arff","/home/joaolucas/CrossSiteScripting.arff","/home/joaolucas/CSRF.arff","/home/joaolucas/DenialOfService.arff","/home/joaolucas/DirectoryTraversal.arff","/home/joaolucas/Doubt.arff","/home/joaolucas/ExecuteCode.arff","/home/joaolucas/FileInclusion.arff","/home/joaolucas/GainInformation.arff","/home/joaolucas/GainPrivileges.arff","/home/joaolucas/InformationDisclosure.arff","/home/joaolucas/InputHandling.arff","/home/joaolucas/MemoryCorruption.arff","/home/joaolucas/ObtainInformation.arff","/home/joaolucas/Overflow.arff","/home/joaolucas/Phishing.arff","/home/joaolucas/RaceCondition.arff","/home/joaolucas/Spoofing.arff","/home/joaolucas/XMLInjection.arff")
dataSet <- c("/home/joaolucas/AuthAlgorithms.arff")

for(file in dataSet){
  print("============================================ J48Pruned ============================================")
  J48Pruned(file)
  
  print("============================================ J48Unpruned ============================================")
  J48Unpruned(file)
  
  print("============================================ J48ReduceError ============================================")
  J48ReduceError(file)
  
  #print("============================================ JRip ============================================")
  #Jrip(file)
  
  print("============================================ NaiveBayes ============================================")
  NaiveBayes(file)
  
  print("============================================ RandomForest ============================================")
  RandomForest(file)
  
  print("============================================ SMOPolyKernel ============================================")
  SMOPolyKernel(file)
  
  print("============================================ SMORBFKernel ============================================")
  SMORBFKernel(file)
  
  print("============================================ C-SVMRadial ============================================")
  cSVMRadial(file)
  
  print("============================================ C-SVMPoly ============================================")
  cSVMPoly(file)
  
  #http://stats.stackexchange.com/questions/37669/libsvm-reaching-max-number-of-iterations-warning-and-cross-validation
  print("============================================ C-SVMLinear ============================================")
  cSVMLinear(file)
  
  print("============================================ C-SVMSigmoid ============================================")
  cSVMSigmoid(file)

  print("============================================ v-SVMRadial ============================================")
  vSVMRadial(file)
  
  print("============================================ v-SVMPoly ============================================")
  vSVMPoly(file)
  
  #http://stats.stackexchange.com/questions/37669/libsvm-reaching-max-number-of-iterations-warning-and-cross-validation
  print("============================================ v-SVMLinear ============================================")
  vSVMLinear(file)
  
  print("============================================ v-SVMSigmoid ============================================")
  vSVMSigmoid(file)
}

