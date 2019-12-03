#installing important packages and the libraries
#install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
library(gbm)
library(randomForest)
library(mlbench)
library(caret)
library(e1071)
library('RANN')
library (mlbench)
data(scat)

#1.Set the Species column as the target/outcome and convert it to numeric. (5 points)
scat$Species<-ifelse(scat$Species=='coyote',0,ifelse(scat$Species=='bobcat',1,2))


#2.Remove the Month, Year, Site, Location features. (5 points)
drop_features <- names(scat) %in% c("Month", "Year", "Site","Location")
descriptive_features <- scat[!drop_features]

#3.Check if any values are null. 
#If there are, impute missing values using KNN. (10 points)
target <- names(descriptive_features) %in% c("Species")
target_features <- descriptive_features[target]
descriptive_features <- descriptive_features[!target]

sum(is.na(descriptive_features))
preProcValues <- preProcess(descriptive_features, method = c("knnImpute"))
train_processed <- predict(preProcValues, descriptive_features)

train_processed<-cbind(target_features, train_processed)

sum(is.na(train_processed))


#4.Converting every categorical variable to numerical (if needed). (5 points)
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))


#Converting the dependent variable back to categorical
train_transformed$Species<-as.factor(train_transformed$Species)
str(train_transformed)
#5.With a seed of 100, 75% training, 25% testing.
set.seed(100)
index <- createDataPartition(train_transformed$Species, p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]

# Build the following models: randomforest, neural 
# net, naive bayes and GBM.
outcomeName<-'Species'

predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
predictors

model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')

################################ GBM ################################
# model summarization 
print(model_gbm)
# plot variable of importance, for the predictions (use the prediction set) display 
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
# confusion matrix
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
#gbm_df<-data.frame(Experiment_name="GBM",Accuracy=postResample(pred = predictions, obs = testSet[,outcomeName])[1],Kappa=postResample(pred = predictions, obs = testSet[,outcomeName])[2])

################################ RANDOM FOREST ################################
# model summarization 
print(model_rf)
# plot variable of importance, for the predictions (use the prediction set) display 
plot(varImp(object=model_rf),main="RF - RANDOM FOREST")
# confusion matrix
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
#rf_df<-data.frame(Experiment_name="RF",Accuracy=postResample(pred = predictions, obs = testSet[,outcomeName])[1],Kappa=postResample(pred = predictions, obs = testSet[,outcomeName])[2])

################################ NEURAL NETWORK ################################
# model summarization
print(model_nnet)
# plot variable of importance, for the predictions (use the prediction set) display 
# Since variable importance of the neural network had outcomes for each class and overall 
# outcome we are ploting variable importance for only overall outcome.
overall=varImp(model_nnet)
df<-(data.frame(values=overall$importance[1:14,1:0]))
df<-cbind(Predictor = rownames(df),df)
barplot(df$values, names = df$Predictor,las=2, main="NNET - NEURAL NETWORK")
grid(nx=NULL, ny=NULL)
# confusion matrix
predictions<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
#NNET_df<-data.frame(Experiment_name="NNET",Accuracy=postResample(pred = predictions, obs = testSet[,outcomeName])[1],Kappa=postResample(pred = predictions, obs = testSet[,outcomeName])[2])

################################ NAIVE BAYES ################################
# model summarization 
print(model_nb)
# plot variable of importance, for the predictions (use the prediction set) display 
plot(varImp(object=model_nb),main="NB - NAIVE BAYES")
# confusion matrix
predictions<-predict.train(object=model_nb,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
#postResample(pred = predictions, obs = testSet[,outcomeName])
#NB_df<-data.frame(Experiment_name="NB",Accuracy=postResample(pred = predictions, obs = testSet[,outcomeName])[1],Kappa=postResample(pred = predictions, obs = testSet[,outcomeName])[2])



# 6.For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) create
# and display a data frame that has the following columns: ExperimentName, accuracy, kappa.
# Sort the data frame by accuracy. (15 points)

gbm_df<-data.frame(Experiment_name="GBM",Accuracy=max(model_gbm$results$Accuracy),Kappa=max(model_gbm$results$Kappa))
rf_df<-data.frame(Experiment_name="RF",Accuracy=max(model_rf$results$Accuracy),Kappa=max(model_rf$results$Kappa))
NNET_df<-data.frame(Experiment_name="NNET",Accuracy=max(model_nnet$results$Accuracy),Kappa=max(model_nnet$results$Kappa))
NB_df<-data.frame(Experiment_name="NB",Accuracy=max(model_nb$results$Accuracy),Kappa=max(model_nb$results$Kappa))

Compare_models_df <- rbind(gbm_df,rf_df,NNET_df,NB_df)

Compare_models_df<-Compare_models_df[order(Compare_models_df$Accuracy),]
rownames(Compare_models_df) <- NULL
Compare_models_df

# 7.Tune the GBM model using tune length = 20 and: a) print the model summary and b) plot the
# models. (20 points)

fitControl <- trainControl(  method = "repeatedcv",  number = 5,  repeats = 5)
model_gbm_tuned<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
plot(model_gbm_tuned)

# 8.Using GGplot and gridExtra to plot all variable of importance plots into one single plot. (10
# points)

#creating DF for all the class of the Naive Bayes
nb_overall=varImp(object=model_nb)
nb_var_df=data.frame(features=rownames(nb_overall$importance),NB_X0=nb_overall$importance$X0,NB_X1=nb_overall$importance$X1,NB_X2=nb_overall$importance$X2)
print("Naive Bayes variable importance Data frame")
nb_var_df

#creating DF for all the class of the Naive Bayes
nnet_overall=varImp(model_nnet)
nnet_var_df<-(data.frame(NNET_values=nnet_overall$importance[1:14,1:0]))
nnet_var_df<-cbind(features = rownames(nnet_var_df),nnet_var_df)

print("Neural Network variable importance Data frame")
nnet_var_df

#Merging in All_model_varImp Data frame
All_model_varImp <- merge(nb_var_df, nnet_var_df, by.x = "features")
print("All Model variable importance Data frame")
All_model_varImp

#creating DF for all the class of the Random Forest
rf_overall=(varImp(object=model_rf))
rf_var_df=data.frame(rf_values=rf_overall$importance)
rf_var_df<-cbind(features = rownames(rf_var_df),RF_values=rf_var_df)

print("Randome Forest variable importance Data frame")
rf_var_df

#Merging in All_model_varImp Data frame
All_model_varImp <- merge(All_model_varImp, rf_var_df, by.x = "features")
colnames(All_model_varImp)[which(names(All_model_varImp) == "Overall")] <- "rf_values"
print("All Model variable importance Data frame")
All_model_varImp

#creating DF for all the class of the GBM
gbm_overall=(varImp(object=model_gbm))
gbm_var_df=data.frame(gbm_values=gbm_overall$importance)
gbm_var_df<-cbind(features = rownames(gbm_var_df),GBM_values=gbm_var_df)
print("GBM variable importance Data frame")
gbm_var_df

#Merging in All_model_varImp Data frame
All_model_varImp <- merge(All_model_varImp, gbm_var_df, by.x = "features")
colnames(All_model_varImp)[which(names(All_model_varImp) == "Overall")] <- "gbm_values"
print("All model variable importance Data frame")
All_model_varImp


# Plot for NNET
nnet_plot<-ggplot(data=All_model_varImp, aes(y=All_model_varImp$NNET_values, x=All_model_varImp$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("NNET Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot for Naive Bayes class 0
NB_X0_plot<-ggplot(data=All_model_varImp, aes(y=All_model_varImp$NB_X0, x=All_model_varImp$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Naive Bayes X0 Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot for Naive Bayes class 1
NB_X1_plot<-ggplot(data=All_model_varImp, aes(y=All_model_varImp$NB_X1, x=All_model_varImp$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Naive Bayes X1 Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot for Naive Bayes class 2
NB_X2_plot<-ggplot(data=All_model_varImp, aes(y=All_model_varImp$NB_X2, x=All_model_varImp$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Naive Bayes X2 Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot for Random Forest
rf_plot<-ggplot(data=All_model_varImp, aes(y=All_model_varImp$rf_values, x=All_model_varImp$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Random forest Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot for GBM
gbm_plot<-ggplot(data=All_model_varImp, aes(y=All_model_varImp$gbm_values, x=All_model_varImp$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("GBM Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Making a grid plot
gridPlot=grid.arrange(gbm_plot,rf_plot,nnet_plot,NB_X0_plot,NB_X1_plot,NB_X2_plot, ncol= 2 ,nrow=3)

gridPlot

# 9.Which model performs the best? and why do you think this is the case? Can we accurately
# predict species on this dataset? (10 points)

# Ans : From the above results of model comparison we can see random forest has maximum accuracy 
# hence we can expect Random forest to perform better than others.
# RF performs best here becasue the provision of feature importance is much more reliable
# when compared to any other model. Also random forest deals with multiple decision tress 
# making it more accurate.
#It can predict Species column 7 out of 10 times approximately as it is evident from the accuracy



# 10.a. Using feature selection with rfe in caret and the repeatedcv method: Find the top 3
# predictors and build the same models as in 6 and 8 with the same parameters. 

#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)


Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
Loan_Pred_Profile

predictors<-c("CN", "d15N", "d13C")

# For example, to apply, GBM, Random forest, Neural net:
model_gbm_featured<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf_featured<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet_featured<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_nb_featured<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')


################################ GBM ################################

#model summarization
print(model_gbm_featured)
#Plotting Varianle importance for GBM
plot(varImp(object=model_gbm_featured),main="GBM - Variable Importance")
predictions<-predict.train(object=model_gbm_featured,testSet[,predictors],type="raw")
table(predictions)
#Confusion Matrix and Statistics
confusionMatrix(predictions,testSet[,outcomeName])

################################ Random Forest ################################

# model summarization
print(model_rf_featured)

# plot variable of importance, for the predictions (use the prediction set) display 
plot(varImp(object=model_rf_featured),main="RF - Random Forest")

#Confusion Matrix and Statistics
predictions<-predict.train(object=model_rf_featured,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])

################################ Neural Network ################################

# model summarization
print(model_nnet_featured)

# plot variable of importance, for the predictions (use the prediction set) display 
overall=varImp(model_nnet_featured)
df_featured<-(data.frame(overall=overall$importance[1:3,1:0]))
df_featured<-cbind(Predictor = rownames(df_featured),df_featured)
barplot(df_featured$overall, names = df_featured$Predictor,las=2, main='NNET - Neural Network')
grid(nx=NULL, ny=NULL)

#Confusion Matrix and Statistics
predictions<-predict.train(object=model_nnet_featured,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])

################################ Naive Bayes ################################

# model summarization
print(model_nb_featured)

# plot variable of importance, for the predictions (use the prediction set) display 
plot(varImp(object=model_nb_featured),main='NB - Naive Bayes')

# Confusion Matrix and Statistics
predictions<-predict.train(object=model_nb_featured,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])




#Using GGplot and gridExtra to plot all variable of importance plots into one single plot.

#Tuning GBM
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
model_gbm_tuned<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
plot(model_gbm_tuned)

#creating DF for all the class of the Naive Bayes
nb_overall_featured=varImp(object=model_nb_featured)
nb_var_df_featured=data.frame(features=rownames(nb_overall_featured$importance),NB_X0_featured=nb_overall_featured$importance$X0,NB_X1_featured=nb_overall_featured$importance$X1,NB_X2_featured=nb_overall_featured$importance$X2)
print("Neural Network variable importance Data frame")
nb_var_df_featured

#creating DF for all the class of the Neural Network
nnet_overall_featured=varImp(model_nnet_featured)
nnet_var_df_featured<-(data.frame(NNET_values_featured=nnet_overall_featured$importance[1:3,1:0]))
nnet_var_df_featured<-cbind(features = rownames(nnet_var_df_featured),nnet_var_df_featured)
print("Neural Network variable importance Data frame")
nnet_var_df_featured

All_model_varImp_featured <- merge(nb_var_df_featured, nnet_var_df_featured, by.x = "features")
All_model_varImp_featured

#creating DF for all the class of the Random forest
rf_overall_featured=(varImp(object=model_rf_featured))
rf_var_df_featured=data.frame(rf_values_featured=rf_overall_featured$importance)
rf_var_df_featured<-cbind(features = rownames(rf_var_df_featured),RF_values=rf_var_df_featured)
print("Random forest variable importance Data frame")
rf_var_df_featured

All_model_varImp_featured <- merge(All_model_varImp_featured, rf_var_df_featured, by.x = "features")
colnames(All_model_varImp_featured)[which(names(All_model_varImp_featured) == "Overall")] <- "rf_values_featured"

All_model_varImp_featured
#creating DF for all the class of the GBM
gbm_overall_featured=(varImp(object=model_gbm_featured))
gbm_var_df_featured=data.frame(gbm_values_featured=gbm_overall_featured$importance)
gbm_var_df_featured<-cbind(features = rownames(gbm_var_df_featured),GBM_values=gbm_var_df_featured)
print("Random forest variable importance Data frame")
gbm_var_df_featured


All_model_varImp_featured <- merge(All_model_varImp_featured, gbm_var_df_featured, by.x = "features")
colnames(All_model_varImp_featured)[which(names(All_model_varImp_featured) == "Overall")] <- "gbm_values_featured"

All_model_varImp_featured

#preparing grid plots
nnet_plot_featured<-ggplot(data=All_model_varImp_featured, aes(y=All_model_varImp_featured$NNET_values_featured, x=All_model_varImp_featured$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("NNET_featured Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

NB_X0_plot_featured<-ggplot(data=All_model_varImp_featured, aes(y=All_model_varImp_featured$NB_X0_featured, x=All_model_varImp_featured$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Naive Bayes X0_featured Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

NB_X1_plot_featured<-ggplot(data=All_model_varImp_featured, aes(y=All_model_varImp_featured$NB_X1_featured, x=All_model_varImp_featured$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Naive Bayes X1_featured Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

NB_X2_plot_featured<-ggplot(data=All_model_varImp_featured, aes(y=All_model_varImp_featured$NB_X2_featured, x=All_model_varImp_featured$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Naive Bayes X2_featured Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

rf_plot_featured<-ggplot(data=All_model_varImp_featured, aes(y=All_model_varImp_featured$rf_values_featured, x=All_model_varImp_featured$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("Random forest_featured Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

gbm_plot_featured<-ggplot(data=All_model_varImp_featured, aes(y=All_model_varImp_featured$gbm_values_featured, x=All_model_varImp_featured$features)) +
  geom_bar(stat="identity",fill="steelblue")+ggtitle("GBM_featured Variable Importance") +
  xlab("Features") + ylab("Values") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

gridPlot_featured=grid.arrange(gbm_plot_featured,rf_plot_featured,nnet_plot_featured,NB_X0_plot_featured,NB_X1_plot_featured,NB_X2_plot_featured, ncol= 2 ,nrow=3)

gridPlot_featured

# 10.b. Create a dataframe that compares the non-feature selected models ( the same as on 7)
# and add the best BEST performing models of each (randomforest, neural net, naive
# bayes and gbm) and display the data frame that has the following columns:
# ExperimentName, accuracy, kappa. Sort the data frame by accuracy. (40 points)

gbm_df_featured<-data.frame(Experiment_name="GBM_featured",Accuracy=max(model_gbm_featured$results$Accuracy),Kappa=max(model_gbm_featured$results$Kappa))
rf_df_featured<-data.frame(Experiment_name="RF_featured",Accuracy=max(model_rf_featured$results$Accuracy),Kappa=max(model_rf_featured$results$Kappa))
NNET_df_featured<-data.frame(Experiment_name="NNET_featured",Accuracy=max(model_nnet_featured$results$Accuracy),Kappa=max(model_nnet_featured$results$Kappa))
NB_df_featured<-data.frame(Experiment_name="NB_featured",Accuracy=max(model_nb_featured$results$Accuracy),Kappa=max(model_nb_featured$results$Kappa))

Compare_models_df_featured <- rbind(gbm_df_featured,rf_df_featured,NNET_df_featured,NB_df_featured)

Compare_models_df_featured<-Compare_models_df_featured[order(Compare_models_df_featured$Accuracy),]

Compare_models_df_featured

Compare_models_all<-rbind(Compare_models_df_featured,Compare_models_df)
Compare_models_all

Compare_models_all<-Compare_models_all[order(Compare_models_all$Accuracy),]
rownames(Compare_models_all) <- NULL
Compare_models_all

# 10.c. Which model performs the best? and why do you think this is the case? Can we
# accurately predict species on this dataset? (10 points)


# Ans : From the above results of model comparison we can see Naïve Bayes Feature selected has maximum accuracy hence we can expect Naïve Bayes Feature selected 
# to perform better than others model. Naïve Bayes Feature selected performs best here because Naïve Bayes works best with continuous data and when the data is small. 
# So if you need something fast we can easily apply and get better results. 
# It can predict Species column 8 out of 10 times approximately as it is evident from the accuracy

