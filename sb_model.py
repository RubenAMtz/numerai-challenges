#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, tree, svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.externals import joblib

def main():
    # Set seed for reproducibility
    seed = np.random.seed(42)

    print("# Loading data...")
    train = pd.read_csv('numerai_training_data.csv', header=0)
    selected_features = pd.read_csv('x_new.csv', header=0)
    tournament = pd.read_csv('numerai_tournament_data.csv', header=0)
    validation = tournament[tournament['data_type']=='validation']

    train_bernie = train

    features = [f for f in list(selected_features) if "feature" in f]

    X = train_bernie[features]
    Y = train_bernie['target_bernie']
    x_prediction = validation[features]

    ids = tournament['id']

    #CONFIGURE YOUR MODELS:
    modelETC = tree.ExtraTreeClassifier()
    print("# Training...")
    modelETC.fit(X, Y)

    modelSGDC = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=0.001)
    print("# Training...")
    modelSGDC.fit(X, Y)

    #Stochastic Gradient Boosting Classification
    num_trees = 100
    kfold = model_selection.KFold(n_splits=len(train['era'].unique()), random_state=seed)
    #Configure model
    modelGBC = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed, verbose=2)
    #Train and test with kfold model iterations
    #results = model_selection.cross_val_score(modelGBC, X, Y, cv=kfold)
    #print(results.mean())
    
    #COMMENT IF YOU DON'T WANT TO SAVE THE TRAINED MODEL
    joblib.dump(modelGBC, 'gradient_classifier.joblib') 
    #UNCOMMENT IF WANT TO LOAD THE TRAINED MODEL
    # modelGBC = joblib.load('gradient_classifier.joblib')
    modelGBC.fit(X,Y)  


    # modelKNN = KNeighborsClassifier(n_neighbors=3)
    # print("# Training...")
    # modelKNN.fit(X, Y)

    #APPLY A CROSS-VALIDATION STRATEGY, SET SCORING TO A COMMON METRIC = MEAN SQUARED ERROR. DEFINE 10 FOLDS
    # print("# Predicting...MODEL ExtraTreeClassifier")
    # scores = cross_val_score(modelETC, X, Y, scoring="neg_mean_squared_error", cv=10)
    # rmse_scores = np.sqrt(-scores)
    # print("Scores:",  rmse_scores)
    # print("Mean:",  rmse_scores.mean())
    # print("Standard deviation:",  rmse_scores.std())

        # print("# Predicting...MODEL Stochastic Gradient Descent Classifier")
        # scores = cross_val_score(modelSGDC, X, Y, scoring="neg_mean_squared_error", cv=10)
        # rmse_scores = np.sqrt(-scores)
        # print("Scores:",  rmse_scores)
        # print("Mean:",  rmse_scores.mean())
        # print("Standard deviation:", rmse_scores.std())

    # print("# Predicting...MODEL K Neighbors Classifier")
    # scores = cross_val_score(modelKNN, X, Y, scoring="neg_mean_squared_error", cv=10)
    # rmse_scores = np.sqrt(-scores)
    # print("Scores:",  rmse_scores)
    # print("Mean:",  rmse_scores.mean())
    # print("Standard deviation:",  rmse_scores.std())
    
    #USED TRAINED MODELS AND TEST THEM AGAINST THE TEST SET (x_prediction is the validation set)
    #print(".")
    # y_prediction = modelETC.predict_proba(x_prediction)
    # probabilities = y_prediction[:, 1]
    # print(probabilities)
    # print("- probabilities ETC:", probabilities[1:6])
    # print("- target:", validation['target_bernie'][1:6])
    # print("- rounded probability:", [round(p) for p in probabilities][1:6])
    # correct = [round(x)==y for (x,y) in zip(probabilities, validation['target_bernie'])]
    # print("- accuracy: ", sum(correct)/float(validation.shape[0]))
    # print("- validation logloss:", metrics.log_loss(validation['target_bernie'], probabilities))
    print(".")
    print("--------")
    print(".")
    y_prediction = modelGBC.predict_proba(x_prediction)
    probabilities = y_prediction[:, 1]
    print(probabilities)
    print("- probabilities SGDC:", probabilities[1:6])
    print("- target:", validation['target_bernie'][1:6])
    print("- rounded probability:", [round(p) for p in probabilities][1:6])
    correct = [round(x)==y for (x,y) in zip(probabilities, validation['target_bernie'])]
    print("- accuracy: ", sum(correct)/float(validation.shape[0]))
    print("- validation logloss:", metrics.log_loss(validation['target_bernie'], probabilities))
    print(".")
    print("--------")
    print(".")
    # y_prediction = modelKNN.predict_proba(x_prediction)
    # probabilities = y_prediction[:, 1]
    # print(probabilities)
    # print("- probabilities KNN:", probabilities[1:6])
    # print("- target:", validation['target_bernie'][1:6])
    # print("- rounded probability:", [round(p) for p in probabilities][1:6])
    # correct = [round(x)==y for (x,y) in zip(probabilities, validation['target_bernie'])]
    # print("- accuracy: ", sum(correct)/float(validation.shape[0]))

    # # The targets for each of the tournaments are very correlated.
        # print("COMPARING AGINST OTHER TARGETS: ----")
        # print("Correlation between targets\n")
        # tournament_corr = np.corrcoef(validation['target_bernie'], validation['target_elizabeth'])
        # print("- bernie vs elizabeth corr:", tournament_corr)
        # # # You can see that target_elizabeth is accurate using the bernie model as well.
        # print("\nRounded probabilities (from original target) vs Elizabeth Target\n")
        # correct = [round(x)==y for (x,y) in zip(probabilities, validation['target_elizabeth'])]
        # print("- elizabeth using bernie:", sum(correct)/float(validation.shape[0]))

    # # Numerai measures models on logloss instead of accuracy. The lower the logloss the better.
    # # Numerai only pays models with logloss < 0.693 on the live portion of the tournament data.
    # # Our validation logloss isn't very good.
    # print("- validation logloss:", metrics.log_loss(validation['target_bernie'], probabilities))

    # # To submit predictions from your model to Numerai, predict on the entire tournament data.
        # print(".")
        # print("PREDICTIONS FOR THE TOURNAMENT *******************")
        # print(".")

        # x_prediction = tournament[features]
    # print("PREDICTIONS USING TREE")
    # y_prediction = modelETC.predict_proba(x_prediction)
    # results = y_prediction[:, 1]
    # results_ETC = pd.DataFrame(data={'probability':results})
    # joined = pd.DataFrame(ids).join(results_ETC)
    # print("- joined:", joined.head())

        # print("\nPREDICTIONS USING SGDC")
        # y_prediction = modelSGDC.predict_proba(x_prediction)
        # results = y_prediction[:, 1]
        # #results = np.round_(results)
        # results_SGDC = pd.DataFrame(data={'probability_bernie':results})
        # joined = pd.DataFrame(ids).join(results_SGDC)
        # print("- joined:", joined.head())
        # print("# Writing predictions to bernie_submissions.csv...")
        # # Save the predictions out to a CSV file.
        # # print("# Creating submission...")
        # joined.to_csv("bernie_submission.csv", index=False)`


    # # Now you can upload these predictions on https://numer.ai

"""
TIPS TO IMPROVE YOUR MODEL

1. Use eras
In this example, we dropped era column but you can use the era column to improve peformance across eras
You can take a model like the above and use it to generate probabilities on the training data, and
look at the the eras where your model was <0.693 and then build a new model on those bad eras to
combine with your main model. In this way, you may be hedged to the risk of bad eras in the future.
Advanced tip: To take this further, you could add the objective of doing consistenty well across all eras
directly into the objective function of your machine learning model.

2. Use feature importance
As per above, you don't want your model to rely too much on any particular type of era. Similarly, you
don't want your model to rely too much on any particular type of feature. If your model relies heavily on
one feature (in linear regression, some feature has very high coefficient), then if that feature doesn't work
in a particular era then your model will perform poorly. If your model balances its use of features then it is
more likely to be consistent across eras.

3. Use all the targets
As we saw above, a model trained on one target like target_bernie might be good at predicting another target
like target_elizabeth. Blending models built on each target could also improve your logloss and consistency.
"""

if __name__ == '__main__':
    main()
