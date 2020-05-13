# Loading libraries
lib_load <- function(){
  
  load.lib = c('tidyverse', "caret", "tidytext", "tm", "SnowballC", "xgboost", 'twitteR',
               'ROAuth', 'tidyverse', 'text2vec', 'glmnet', 'ggrepel', 'textclean', 'MLmetrics')
  #We create a variable where we indicating that we select the ones that we do not have installed.                     
  install.lib <- load.lib[!load.lib %in% installed.packages()]
  #With a loop we install the libraries that we do not have.
  for(lib in install.lib) install.packages(lib,dependences=TRUE)
  #Load the libraries
  sapply(load.lib,require,character=TRUE)
}
lib_load()

# Loading the datasets
train_data = read.csv('updated_train.csv', stringsAsFactors = FALSE)
test_data = read.csv('updated_test.csv', stringsAsFactors = FALSE)
ss = read.csv('updated_ss.csv')

# data splitting on train and validation set
set.seed(2340)
trainIndex <- createDataPartition(train_data$target, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

training_set <- train_data[trainIndex, ]
valid_set <- train_data[-trainIndex, ]

##### doc2vec #####
# define preprocessing function and tokenization function
prep_fun <- tolower
tok_fun <- word_tokenizer

it_training_set <- itoken(training_set$text, 
                          preprocessor = prep_fun, 
                          tokenizer = tok_fun,
                          ids = training_set$ID,
                          progressbar = TRUE)
it_valid_set <- itoken(valid_set$text, 
                       preprocessor = prep_fun, 
                       tokenizer = tok_fun,
                       ids = valid_set$ID ,
                       progressbar = TRUE)

# creating vocabulary and document-term matrix
vocab <- create_vocabulary(it_training_set)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_training_set, vectorizer)
dtm_valid <- create_dtm(it_valid_set, vectorizer)

# define tf-idf model
tfidf <- TfIdf$new()

# fit the model to the train data and transform it with the fitted model
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_valid_tfidf <- fit_transform(dtm_valid, tfidf)

# train the model - Model 1 - cv.glmnet model
t1 <- Sys.time()
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = training_set[['target']], 
                               family = 'binomial', 
                               # L1 penalty
                               alpha = 1,
                               # interested in the area under ROC curve
                               type.measure = "auc",
                               # 5-fold cross-validation
                               nfolds = 10,
                               # high value is less accurate, but has faster training
                               thresh = 1e-3,
                               # again lower number of iterations for faster training
                               maxit = 1e3,
                               relax = TRUE)

print(difftime(Sys.time(), t1, units = 'mins'))

plot(glmnet_classifier)

preds <- predict(glmnet_classifier, dtm_valid_tfidf, type = 'response')[ ,1]

glmnet:::auc(as.numeric(valid_set$target), preds)

#Finding log loss (The metric of evaluation in the competition)
LogLoss(preds, valid_set$target) # This model gave a logloss of 0.31

# save the model for future using
saveRDS(glmnet_classifier, 'glmnet_classifier.RDS')

# train the model - Model 2 - XGBoost Model
xg_classifier = xgboost(data = dtm_train_tfidf, label = training_set[['target']], objective = "binary:logistic",nrounds = 10,
                        max.depth = 30, eta = 0.5, nthread = 10, gamma = 0.9)

y_pred = predict(xg_classifier, newdata = dtm_valid_tfidf, type = 'response')
LogLoss(y_pred, valid_set$target) # This model gave a logloss of 0.26


### Working on the new test data
# preprocessing and tokenization
it_tweets <- itoken(test_data$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = test_data$ID,
                    progressbar = TRUE)

# creating vocabulary and document-term matrix
dtm_tweets <- create_dtm(it_tweets, vectorizer)

# transforming data with tf-idf
dtm_tweets_tfidf <- fit_transform(dtm_tweets, tfidf)

# predict probabilities of the new test data for submission using cv.glmnet model
preds_tweets <- predict(glmnet_classifier, dtm_tweets_tfidf, type = 'response')

subfile = as.data.frame(preds_tweets)

# Writing a csv file for submission
write.csv(subfile, "Subfile2.csv", row.names = TRUE) # Returned a Log-Loss of 0.30


# predict probabilities of the new test data for submission using XGBoost model
preds_tweets <- predict(xg_classifier, dtm_tweets_tfidf, type = 'response')
ss$target = preds_tweets
head(ss)

# Writing a csv file for submission
write.csv(ss, "Subfile2xg.csv", row.names = FALSE) # Returned a Log-Loss of 0.27


