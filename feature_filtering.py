"""
Creates a csv file based on the selection of features using ExtraTreesClassifier
"""
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def main():
    
    np.random.seed(42)
    
    train = pd.read_csv('./datasets/numerai_training_data.csv', header=0)

    train_bernie = train.drop(['id','data_type','target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], axis=1)
    features = [f for f in list(train_bernie) if "feature" in f]
    features = np.array(features)

    X = train_bernie[features]
    y = train_bernie['target_bernie']
    
    clf = ExtraTreesClassifier(n_estimators=70)
    clf = clf.fit(X, y)

    importances = clf.feature_importances_
    importances = importances.tolist()
    index_ordered = np.argsort(importances)[::-1]

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    index = model.get_support([index_ordered])
    
    x_shape = X_new.shape
    X_new = pd.DataFrame(data = X_new, columns = features[index[0:x_shape[1]]])
    X_new['target_bernie'] = y
    X_new.to_csv(path_or_buf="./datasets/x_new.csv", index=False)    

if __name__ == '__main__':
    main()