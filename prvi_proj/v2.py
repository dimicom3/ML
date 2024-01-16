# %%
import itertools
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, auc, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline as make_pipeline
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
# %%
df = pd.read_csv("data.csv")
# %%
df.shape
# %%
df.info()
# %%
df.head(10)
# %%
df.describe()
# %%
df['Bankrupt?'].value_counts()
# %%
df.hist(figsize = (35,30), bins = 50)
plt.show()
# %%
num_features = len(df.select_dtypes(include='number').columns)
num_cols = 9
num_rows = (num_features + num_cols - 1) // num_cols 

f, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(35, num_rows * 6))

for i, col in enumerate(df.select_dtypes(include='number').columns):
    sb.boxplot(x='Bankrupt?', y=col, data=df, ax=axes[i // num_cols, i % num_cols])
    axes[i // num_cols, i % num_cols].set_title(col)

plt.tight_layout()
plt.show()
# %%
num_columns = len(df.columns)
num_plots = (num_columns + 3) // 4

fig, axs = plt.subplots(num_plots, 4, figsize=(20, 5 * num_plots), constrained_layout=True)

for i, col in enumerate(df.columns):
    row, col_num = divmod(i, 4)
    
    ax = axs[row, col_num]
    
    sb.distplot(df[(df['Bankrupt?'] == 0) & (df[col] < 1)][col].apply(lambda x: x * 100),
                color='green', label='Not Bankrupt', ax=ax)
    
    sb.distplot(df[(df['Bankrupt?'] == 1) & (df[col] < 1)][col].apply(lambda x: x * 100),
                color='red', label='Bankrupt', ax=ax)
    
    ax.set_title(f'{col} Distribution for Bankrupt and Not Bankrupt Companies', fontsize=12)
    ax.set_xlabel(f'{col} %', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(loc='upper right')

plt.show()
# %%
[print(col) for col in df if df[col].isna().sum() > 0]
# %%
df.duplicated().sum()
# %%

num_columns = len(df.columns)
num_plots = (num_columns + 3) // 4  # Calculate the number of required plots, rounding up

fig, axs = plt.subplots(num_plots, 4, figsize=(20, 5 * num_plots), constrained_layout=True)

for i, col in enumerate(df.columns):
    row, col_num = divmod(i, 4)
    
    ax = axs[row, col_num]
    
    sb.histplot(df[(df['Bankrupt?'] == 0) & (df[col] < 1)][col].apply(lambda x: x * 100),
                 color='green', label='Not Bankrupt', ax=ax, kde=True, bins=20)
    
    sb.histplot(df[(df['Bankrupt?'] == 1) & (df[col] < 1)][col].apply(lambda x: x * 100),
                 color='red', label='Bankrupt', ax=ax, kde=True, bins=20)
    
    ax.set_title(f'{col} Distribution for Bankrupt and Not Bankrupt Companies', fontsize=12)
    ax.set_xlabel(f'{col} %', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(loc='upper right')

plt.show()

# %%
def outlier_thresholds(dataframe, variable, low_quantile=0.15, up_quantile=0.85):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
# for col in num_cols:
#       col, check_outlier(df, col)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



def outliers_removal(feature,feature_name,dataset):
    
    q25, q75 = np.percentile(feature, 25), np.percentile(feature, 75)
    feat_iqr = q75 - q25
    
    feat_cut_off = feat_iqr * 1.5
    feat_lower, feat_upper = q25 - feat_cut_off, q75 + feat_cut_off

    dataset = dataset.drop(dataset[(dataset[feature_name] > feat_upper) | (dataset[feature_name] < feat_lower)].index)
    
    return dataset

for col in df:
    new_df = outliers_removal(df[col],str(col),df)
# for col in num_cols:
#         replace_with_thresholds(df,col)
    

# %%
corr_matrix = df.corr()
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
img = ax.imshow(corr_matrix, cmap='magma', interpolation='nearest', aspect='auto')
ax.set_xticks(np.arange(len(corr_matrix.columns)), labels=list(corr_matrix.columns))
ax.set_yticks(np.arange(len(corr_matrix.columns)), labels=list(corr_matrix.columns))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.colorbar(img)
plt.show()
# %%
columns_to_drop = [' Liability-Assets Flag', ' Net Income Flag']
df = df.drop(columns=columns_to_drop)
# %%
## REMOVE CORRELATED FEATUERS
corr_ft = set()
corr_threshold = 0.9
for i in range(len(corr_matrix)):
    for j in range(i):
        if((abs(corr_matrix.iloc[i, j]) >= corr_threshold) and (corr_matrix.iloc[i, j] not in corr_ft)):
            corr_ft.add(corr_matrix.columns[i])
ft_num_old = df.shape[1]
df = df.drop(columns=corr_ft)
ft_removed = ft_num_old - df.shape[1]
ft_removed
# %%
X = df.drop('Bankrupt?', axis = 1)
y = df['Bankrupt?']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


X_train,X_test,y_train,y_test  = train_test_split(X_scaled,
                                              y,
                                              test_size=0.2,
                                              stratify = y,
                                              random_state = 42)

#skf = StratifiedKFold(n_splits=5)#, random_state=42, shuffle=True)


# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def print_evaluation_report(y_test, y_pred, y_pred_proba):
    """This function prints the merices involved for evalutions of the model.
        It also calls the confusioj matrix plot function
    """
    
    cm = confusion_matrix(y_test, y_pred)
    #np.set_printoptions(precision=2)
    #print(cm)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print('F1 Score : %.2f' % f1)
    print('AUC ROC : %.2f' % auc)
    print('Precision : %.2f' % precision)
    print('Recall : %.2f' % recall)
    
    classes = [0,1]
    plt.figure()
    plot_confusion_matrix(cm, classes,title = 'Confusion Matrix', normalize = False)
    plt.show()
    return

# %%
rfc =  RandomForestClassifier().fit(X_train, y_train)
y_pred = rfc.predict(X_test)
y_pred_proba = rfc.predict_proba(X_test)
y_pred_proba = y_pred_proba[:, 1]

print_evaluation_report(y_test, y_pred, y_pred_proba)
# %%

# %%
smoteSampler = SMOTE()
X_train_smote, y_train_smote = smoteSampler.fit_resample(X_train, y_train)

# %%
results = pd.DataFrame(
    columns=["Algorithm",  "CV-AUC", "CV-Accuracy", "CV-F1",
             "AUC", "Accuracy", "F1",])
results.set_index('Algorithm', inplace=True)

def cross_validate_models(classifiers, X, y, cv=5, results=None):
    for name in classifiers:
        classifier = classifiers[name]
        scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'recall']
        cv_results = cross_validate(
            classifier, X, y, cv=cv, scoring=scoring_metrics)
        acc = cv_results['test_accuracy'].mean()
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        recall = cv_results['test_recall'].mean()
        results.loc[name, 'CV-F1'] = f1
        results.loc[name, 'CV-AUC'] = auc
        results.loc[name, 'CV-Accuracy'] = acc
        results.loc[name, 'CV-Recall'] = recall

def test_models(classifiers, X_train, X_test, y_train, y_test, results):
    sb.set_palette('hls')
    confusion_matrix_fig, confusion_matrix_ax =\
        plt.subplots(1, len(classifiers), figsize=(
            15, 4), constrained_layout=True,)
    all_curves_fig, all_curves_ax = plt.subplots(figsize=(10, 10))
    for i, name in enumerate(classifiers):
        classifier = classifiers[name]
        classifier.fit(X_train, y_train)

        if hasattr(classifier, 'predict_proba'):
            y_pred_prob = classifier.predict_proba(X_test)[:, 1]
        elif hasattr(classifier, 'decision_function'):
            decision_function = classifier.decision_function(X_test)
            y_pred_prob = 1 / (1 + np.exp(-decision_function))
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        pred = classifier.predict(X_test)
        f1 = f1_score(y_test, pred)
        acc = accuracy_score(y_test, pred)
        recall = recall_score(y_test, pred)

        results.loc[name, "F1"] = f1
        results.loc[name, "AUC"] = roc_auc
        results.loc[name, "Accuracy"] = acc
        results.loc[name, "Recall"] = recall

        all_curves_ax.plot(fpr, tpr, lw=2,
                           label=f'{name} (AUC = {roc_auc:.2f})')

        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm).plot(
            ax=confusion_matrix_ax[i], colorbar=False, cmap=plt.cm.Blues)
        confusion_matrix_ax[i].grid(False)
        confusion_matrix_ax[i].set_title(name)

    all_curves_ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    all_curves_ax.set_title('All ROC Curves')
    all_curves_ax.set_xlabel('False Positive Rate')
    all_curves_ax.set_ylabel('True Positive Rate')
    all_curves_ax.legend(loc='lower right')
    all_curves_fig.show()
    confusion_matrix_fig.show()
    results.sort_values(by='AUC', ascending=False)
    return results
# %%


classifiers = {
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNNeighbors': KNeighborsClassifier(),
    'SVC': SVC(),
    'XGBoost': XGBClassifier()
}

classifiers_cv = {
    'Random Forest': make_pipeline(SMOTE(random_state=42), RandomForestClassifier()),
    'AdaBoost': make_pipeline(SMOTE(random_state=42), AdaBoostClassifier()),
    'Bagging': make_pipeline(SMOTE(random_state=42), BaggingClassifier()),
    'Logistic Regression': make_pipeline(SMOTE(random_state=42), LogisticRegression(max_iter=1000, random_state=42)),
    'KNNeighbors': make_pipeline(SMOTE(random_state=42), KNeighborsClassifier()),
    'SVC': make_pipeline(SMOTE(random_state=42), SVC()),
    'XGBoost': make_pipeline(SMOTE(random_state=42), XGBClassifier())
}

cross_validate_models(classifiers_cv, X_train,
                      y_train, results=results, cv=5)
results = test_models(classifiers, X_train_smote, X_test, y_train_smote, y_test, results)


# %%
results.sort_values(by='CV-F1', ascending=False)
# %%
rf_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='roc_auc')
rf_grid_search.fit(X_train_smote, y_train_smote)
rf_grid_search.best_params_
#{'max_depth': None,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 200}
# %%
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.2, 0.3],
    'subsample': [1.0, 1.1],
    'colsample_bytree': [0.9, 1.0]
}

xgb_grid_search = GridSearchCV(XGBClassifier(), xgb_param_grid, cv=5, scoring='roc_auc')

xgb_grid_search.fit(X_train_smote, y_train_smote)

print("XGBoost params:", xgb_grid_search.best_params_)
#XGBoost Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200, 'subsample': 1.0}
# %%


hp_optimized_models = {
    'Random Forest (optimized)': RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200),
    'XGBoost (optimized)': XGBClassifier(colsample_bytree = 1.0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0),
}
hp_optimized_models_cv = {
    'Random Forest (optimized)': make_pipeline(SMOTE(random_state=42), RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)),
    'XGBoost (optimized)': make_pipeline(SMOTE(random_state=42), XGBClassifier(colsample_bytree = 1.0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0)),
}
cross_validate_models(hp_optimized_models_cv, X_train,
                      y_train, results=results)
results = test_models(hp_optimized_models, X_train_smote,
                      X_test, y_train_smote, y_test, results)


# %%
results.sort_values(by='CV-F1', ascending=False)

# %%


randomForest = RandomForestClassifier(random_state=0, criterion="entropy")
importances = hp_optimized_models['Random Forest (optimized)'].feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]

plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()


# %%
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_smote)
X_test_pca = pca.transform(X_test)
X_test_pca.shape

pca_optimized_models = {
    'Random Forest (optimized) + PCA': RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200),
    'XGBoost (optimized) + PCA': XGBClassifier(colsample_bytree = 1.0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0),
}

results = test_models(pca_optimized_models, X_train_pca,
                      X_test_pca, y_train_smote, y_test, results)

# %%
results.sort_values(by='F1', ascending=False)

# %%
pca_optimized_models = {
    'Random Forest (optimized) + PCA': RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200),
    'XGBoost (optimized) + PCA': XGBClassifier(colsample_bytree = 1.0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0),
}

results = test_models(pca_optimized_models, X_train_pca,
                      X_test_pca, y_train_smote, y_test, results)

# %%

lasso = Lasso(alpha=0.00001)
lasso.fit(X_train_smote, y_train_smote)
lasso_coef = np.abs(lasso.coef_)
pd_lasso = pd.DataFrame(lasso_coef, index=X.columns, columns=['coef'])

pd_lasso.sort_values(by='coef', ascending=False, inplace=True)
plt.figure(figsize=(15, 10))
plt.xticks(rotation=90)
plt.bar(pd_lasso.index, pd_lasso['coef'])
plt.ylabel("Importance")
plt.title("LASSO Feature Importance")
plt.show()


# %%
fi_optimized_models = {
    'Random Forest (optimized) + fi': RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=500),
    'XGBoost (optimized) + fi': XGBClassifier(colsample_bytree = 1.0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0),
}
selected_features = lasso_coef >= 0.015

X_train_lasso = pd.DataFrame(X_train_smote, columns=X.columns)
X_train_lasso = X_train_lasso.loc[:, selected_features]

X_test_lasso = pd.DataFrame(X_test, columns=X.columns)
X_test_lasso = X_test_lasso.loc[:, selected_features]


results = test_models(fi_optimized_models, X_train_lasso, X_test_lasso, y_train_smote, y_test, results)
# %%
results.sort_values(by='F1', ascending=False)
# %%