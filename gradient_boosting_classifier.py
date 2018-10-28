import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

def main():
    # Set seed for reproducibility
    seed = np.random.seed(42)

    print("# Loading data...")
    train = pd.read_csv('./datasets/numerai_training_data.csv', header=0)
    selected_features = pd.read_csv('./datasets/x_new.csv', header=0)
    tournament = pd.read_csv('./datasets/numerai_tournament_data.csv', header=0)
    validation = tournament[tournament['data_type']=='validation']

    train_bernie = train

    features = [f for f in list(selected_features) if "feature" in f]

    X = train_bernie[features]
    Y = train_bernie['target_bernie']
    x_prediction = validation[features]

    ids = tournament['id']

    #CONFIGURE YOUR MODELS:
    #Stochastic Gradient Boosting Classification
    num_trees = 25
    kfold = model_selection.KFold(n_splits=len(train['era'].unique()), random_state=seed)
    #Configure model
    modelGBC = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed, verbose=2)
    #Train and test with kfold model iterations
    #results = model_selection.cross_val_score(modelGBC, X, Y, cv=kfold)
    #print(results.mean())
    #COMMENT IF YOU DON'T WANT TO SAVE THE TRAINED MODEL
    joblib.dump(modelGBC, './models/gradient_boosting_classifier.joblib') 
    #UNCOMMENT IF WANT TO LOAD THE TRAINED MODEL
    # modelGBC = joblib.load('gradient_classifier.joblib')
    modelGBC.fit(X,Y)  

    
    #USED TRAINED MODELS AND TEST THEM AGAINST THE TEST SET (x_prediction is the validation set)
    y_prediction = modelGBC.predict_proba(x_prediction)
    probabilities = y_prediction[:, 1]
    print(probabilities)
    print("- probabilities GBC:", probabilities[1:6])
    print("- target:\n", validation['target_bernie'][1:6])
    print("- rounded probability:", [round(p) for p in probabilities][1:6])
    correct = [round(x)==y for (x,y) in zip(probabilities, validation['target_bernie'])]
    print("- accuracy: ", sum(correct)/float(validation.shape[0]))
    print("- validation logloss:", metrics.log_loss(validation['target_bernie'], probabilities))

    # # To submit predictions from your model to Numerai, predict on the entire tournament data.
    print("PREDICTIONS FOR THE TOURNAMENT *******************")

    x_prediction = tournament[features]
    print("\nPREDICTIONS USING GBC")
    y_prediction = modelGBC.predict_proba(x_prediction)
    results = y_prediction[:, 1]
    #results = np.round_(results)
    results_GBC = pd.DataFrame(data={'probability_bernie':results})
    joined = pd.DataFrame(ids).join(results_GBC)
    print("- joined:", joined.head())
    print("# Writing predictions to bernie_submissions_gbc.csv...")
    # Save the predictions out to a CSV file.
    # print("# Creating submission...")
    joined.to_csv("./results/bernie_submission_gbc.csv", index=False)


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
