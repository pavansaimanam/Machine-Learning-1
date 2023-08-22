library(caret)
library(gbm)
library(glmnet)
library(xgboost)
library(e1071)
library(splines)
# Set seed for reproducibility
set.seed(123456)
# Import training data
train1 <- read.csv("training_data.csv")
# Extract features and target variable from training data
x_train1 <- train1[, -grep('Y', colnames(train1))]
y_train1 <- train1$Y
# Standardize train dataset
x_train1 <- scale(x_train1)
# Import test data
test1 <- read.csv("test_data.csv")
# Standardize test dataset using training data's mean and standard deviation
test_data <- scale(test1)
# Fit various models to the training data
# Gradient boosting model
gb_model <- train(y = y_train1, x = x_train1, method = "gbm", trControl = fitControl, 
                  tuneGrid = expand.grid(interaction.depth = 1:5, n.trees = (1:10) * 10, 
                                         shrinkage = 0.1, n.minobsinnode = 10))
gb_pred <- predict(gb_model$finalModel, newdata = x_train1)
gb_rmse <- sqrt(mean((gb_pred - y_train1)^2))
# LASSO regression
las <- cv.glmnet(x_train1, y_train1, alpha = 1)
best_las_lambda <- las$lambda.min
lasso_model <- glmnet(x_train1, y_train1, alpha = 1, lambda = best_las_lambda)
lasso_pred <- predict(lasso_model, newdata = x_train1)
lasso_rmse <- sqrt(mean((lasso_pred - y_train1)^2))
# Ridge regression
ridge_model <- glmnet(x_train1, y_train1, alpha = 0)
ridge_pred <- predict(ridge_model, newdata = x_train1)
ridge_rmse <- sqrt(mean((ridge_pred - y_train1)^2))
# XGBoost model
xg_model <- xgboost(data = x_train1, label = y_train1, max.depth = 10, eta = 1, nthread = 3, nrounds = 2)
xg_pred <- predict(xg_model, newdata = x_train1)
xg_rmse <- sqrt(mean((xg_pred - y_train1)^2))
# SVM
svm_model <- svm(y_train1 ~ ., data = as.data.frame(x_train1))
svm_pred <- predict(svm_model, newdata = x_train1)
svm_rmse <- sqrt(mean((svm_pred - y_train1)^2))
# Spline regression for non-linear relationships
x_train1_4 <- x_train1[, 4]
spline_model <- smooth.spline(x = x_train1_4, y = y_train1, df = 10)
spline_pred <- predict(spline_model, newdata = x_train1)
spline_rmse <- sqrt(mean((spline_pred - y_train1)^2))
# Make predictions on test data using the best model (Gradient Boosting)
pred_gb <- predict(gb_model$finalModel, newdata = test_data)
# Write predictions to a CSV file
write.csv(pred_gb, "Predictions_GB.csv", row.names = FALSE)
