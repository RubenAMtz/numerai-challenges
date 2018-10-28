# NumerAI Challenges

NUMERAI is a weekly data science competition.

Predictions submitted by users steer Numerai's hedge fund together.

A weekly prize pool is paid out to the top performing users who stake.

Interested? Here is how to get started.

## Description

STOCKBREAKER

NumerAI releases a weekly challenge for Data Scientists, it is an open and descentralized market place for predictions.
Each new round comes with a unique data zip. This zip includes all the data you will need to compete on Numerai.

Training data - a CSV file containing rows of features and their binary target values. Use this to train your machine learning model.

Tournament data - a CSV file with the same structure containing rows for validation, test and live. Run your trained model against these rows to generate your predictions.

The dataset we provide is unlike what you would find in other data science competitions:

Cleaned and Regularized - we do the prep work so you can focus on building the model.

Obfuscated Data - row IDs and feature names are hidden to remove any human bias from the data science.

Eras represent an unspecified period of time.

## Objective

Submissions are graded on 3 metrics:

### Logarithmic loss - the primary measure of your predictions's performance. The benchmark for log loss is <0.693.

There are 3 types of log loss: Validation logloss is graded against the validation targets and is publically displayed. Test logloss is graded against test targets that is only known to Numerai and is hidden. This gives us an internal estimate of how well we think you will do. Live logloss is calculated at round resolution against live market data, which is used for payout calculation is publically displayed.

### Consistency - the percentage of eras in which your predictions beat the log loss benchmark on the validation set. A submission is considered consistent if consistency >=58%.

Numerai is looking for models that perform well consistently across eras and not just in some. This is important to us because we actually use these predictions to make real trades on the market.

 Having trouble with consistency? Try adding the objective of doing consistently well across all eras directly into the objective function of your model.

### Concordance - whether predictions on the validation, test and live set appear to be produced by the same model. Concordance is required.

 The code to calculate these metrics is open source. Check it out at numerai/submission-criteria.  

 Looking for inspiration? Check out the following guides from users (rules/payouts referenced may be outdated): Numerai walkthrough, Notes on the Numerai ML Competition, How I predicted the stock market at Numerai, Numerai - Like Kaggle but with a clean dataset, and Revisiting Numerai.

## Installation

Instructions for installation and dependencies:

### Dependencies

In order to run this project in your computer you need Python 3 installed and the following packages:  

- pandas
- scikit-learn
- numpy

### Install

Once you have installed all the dependencies, you can proceed to clone (or download) the repository.  

In order to do that, type the following command in your terminal:

```
git clone https://github.com/RubenAMtz/numerai-challenges.git
```

Next you need to cd into the cloned repo:

```
cd numerai-challenges
```

And run `feature_filtering.py` for feature selection.  

Then select either classification model to train with any of the following commands:  

`python gradient_boosting_classifier.py` or `python stochastic_gradient_descent_classifier.py`

## How to participate

Instructions of how to participate can be found in the following link:  
[https://numer.ai/](https://numer.ai/)
