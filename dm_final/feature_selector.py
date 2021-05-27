from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

def get_tree_selector(X, y):
	return SelectFromModel(ExtraTreesClassifier()).fit(X, y)

def get_KB_selector(func, X, y, k):
    if func == "chi2":
        return SelectKBest(chi2, k=k).fit(X, y)
    if func == "f_classif":
        return SelectKBest(f_classif, k=k).fit(X, y)
    if func == "mutual_info_classif":
        return SelectKBest(mutual_info_classif, k=k).fit(X, y)