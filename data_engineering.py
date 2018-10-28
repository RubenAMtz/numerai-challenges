import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def main():
    
    np.random.seed(42)
    
    train = pd.read_csv('numerai_training_data.csv', header=0)

    train_bernie = train.drop(['id','data_type','target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], axis=1)
    features = [f for f in list(train_bernie) if "feature" in f]
    features = np.array(features)

    X = train_bernie[features]
    y = train_bernie['target_bernie']
    print(X.shape)
    clf = ExtraTreesClassifier(n_estimators=70)
    clf = clf.fit(X, y)
    importances = clf.feature_importances_
    importances = importances.tolist()
    # argsort returns the indices that would sort the list, from lowest to greatest,
    # [::-1] reverses the order of a list. To bring the greatest to the first position.
    index_ordered = np.argsort(importances)[::-1]

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    index = model.get_support([index_ordered])
    #print(index)
    x_shape = X_new.shape
    X_new = pd.DataFrame(data = X_new, columns = features[index[0:x_shape[1]]])
    X_new['target_bernie'] = y
    X_new.to_csv(path_or_buf="./x_new.csv", index=False)
    
    print(X_new.shape)
    #print(X_new)
    #print(features[indices])
    #features = train_bernie.iloc[:, indices]
    #print(features)


    
# ?
if __name__ == '__main__':
    main()