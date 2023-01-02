import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, mutual_info_classif, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')

class Classification_Evaluator:
    """ An Evaluator for Classification.

    Test different feature selection methods and compute the performance of each method for each model.

    Attributes
    ----------
    report : pd.DataFrame
        Aggregated performance of each combination of method and model.

    best_settings : dict
        Best combination of model, method and features that result in a best metric.

    Notes
    -----
    Do not allow NaN values in the input.
    Raises ValueError if there is a NaN value in any of the datasets.

    Examples
    --------
    The following dataset has a combination of categorical and numerical columns.
    The categorical columns were preprocessing using OneHotEncoder.
    The dataset were splited using sklearn.model_selection.train_test_split.

        >>> import feature_selection as fs
        >>> eva = fs.Classification_Evaluator()
        >>> eva.add_data(X_train, y_train, X_test, y_test)
        >>> eva.categorical_columns(['country_France', 'country_Germany', 'country_Spain',
                                     'gender_Female', 'gender_Male', 'active_member_0', 'active_member_1'])
        >>> eva.numerical_columns(['credit_score', 'age', 'tenure', 'balance', 'products_number',
                                   'credit_card', 'estimated_salary'])
        >>> eva.evaluate_models()
    """
    
    def __init__(self, metric='accurary'):
        # Setting the dict that will store the features for each method and model
        self.features = {
            'Logistic Regression': {},
            'Naive Bayes': {},
            'Gradient Descent': {},
            'Decision Tree': {},
            'Adaboost': {},
            'Random Forest': {}
        }
        # Setting the dict that will store the metrics for each method and model
        self.metrics = {
            'Logistic Regression': {},
            'Naive Bayes': {},
            'Gradient Descent': {},
            'Decision Tree': {},
            'Adaboost': {},
            'Random Forest': {}
        }
        # Methods that will be used
        self.methods = ['Chi2 Filter', 'Variance Filter', 'Info Gain Filter',
                        'Forward Selection', 'Backward Selection', 'Lasso Importance',
                        'RF Importance', 'Chosen by One', 'Chosen by All']
        # Setting the metric that will be used
        self.metric_name = metric
        if metric == 'accurary':
            self.metric = accuracy_score
        elif metric == 'precision':
            self.metric = precision_score
        elif metric == 'recall':
            self.metric = recall_score
        elif metric == 'f1':
            self.metric = f1_score
        elif metric == 'roc_auc':
            self.metric = roc_auc_score
        elif metric == 'average_precision':
            self.metric = average_precision_score
        
    def add_data(self, X_train, y_train, X_test, y_test):
        """Add the train and test dataset for futher use.

        Parameters:
            X_train (pd.DataFrame): Train dataset with predictors variables.
            y_train (pd.DataFrame): Train dataset with target variables.
            X_test (pd.DataFrame): Test dataset with predictors variables.
            y_test (pd.DataFrame): Test dataset with target variables.
        """

        # Checking if there is null values in any dataset
        assert X_train.isnull().sum().sum() == 0, 'Há nulos em X_train, preencher ou remover os nulos'
        assert y_train.isnull().sum() == 0, 'Há nulos em y_train, preencher ou remover os nulos'
        assert X_test.isnull().sum().sum() == 0, 'Há nulos em X_test, preencher ou remover os nulos'
        assert y_test.isnull().sum() == 0, 'Há nulos em y_test, preencher ou remover os nulos'
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def categorical_columns(self, cat_cols):
        """ Specify which columns on the dataset are categorical.

        Parameters:
            cat_cols (list): List with the categorical columns names.
        """

        self.cat_col = cat_cols
        
    def numerical_columns(self, num_cols):
        """ Specify which columns on the dataset are numerical.

        Parameters:
            num_cols (list): List with the numerical columns names.
        """

        self.num_col = num_cols
        
    def chi2_filter(self):
        """ Apply Chi2 test on categorical features to filter out only the most valuable ones."""

        chi2_feat = SelectKBest(chi2, k=3)
        chi2_feat.fit_transform(self.X_train[self.cat_col], self.y_train)
        self.chi2_features = chi2_feat.get_feature_names_out()
        
    def variance_filter(self):
        """ Apply Variance filter that filter out numerical features with low variance."""

        v_feat = VarianceThreshold()
        v_feat.fit(self.X_train[self.num_col])
        self.var_features = v_feat.get_feature_names_out()
        
    def information_filter(self):
        """ Apply Information Value and filter out features that the value is more than zero"""

        importances = mutual_info_classif(self.X_train, self.y_train)
        # Format a dataframe with features and their information values
        importances = (
            pd.DataFrame(
                np.hstack(
                    (np.array(self.X_train.columns).reshape((14, 1)),
                     np.array(importances).reshape((14, 1)))
                ),
                columns=['features', 'importances']
            )
        )
        # Filtering out features that the values are more than zero
        self.info_features = importances[importances['importances']!=0]['features'].values        

    def ffs_wrapper(self, model):
        """ Use Forward Feature Selection to select the best set of features.

        Returns:
            feature_names (list): Names of the best set of features.
        """

        ffs = SequentialFeatureSelector(model, k_features='best', forward=True, n_jobs=-1)
        ffs.fit(self.X_train, self.y_train)
        return list(ffs.k_feature_names_)
    
    def bfs_wrapper(self, model):
        """ Use Backward Feature Selection to select the best set of fetures.

        Returns:
            feature_names (list): Names of the best set of features.
        """

        bfs = SequentialFeatureSelector(model, k_features='best', forward=False, n_jobs=-1)
        bfs.fit(self.X_train, self.y_train)
        return list(bfs.k_feature_names_)
    
    def lasso_embedded(self):
        """ Apply L1 Regularization to filter out the best set of features."""

        lr = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=79)
        lasso = SelectFromModel(lr).fit(self.X_train, self.y_train)
        self.lasso_features = lasso.get_feature_names_out()
        
    def rf_embedded(self):
        """ Uses the Feature Importance from a Random Forest to filter out some low importance features."""

        # Train the RF
        rf = RandomForestClassifier(n_estimators=100, random_state=79)
        rf.fit(self.X_train, self.y_train)

        # Set a dataframe with feature names and their importance
        importances = rf.feature_importances_
        importances =pd.DataFrame({'Features': self.X_train.columns, 'Importances': importances})
        
        # Filter only the features that their importance are more than 5% of the highest importance value
        thresh = importances['Importances'].max() * 0.05
        self.rf_features = importances[importances['Importances']>=thresh]['Features'].values
    
    def train_model(self, model, features):
        """ Train the giving model using the giving features.

        Parameters:
            model : Model.
            features (list): Set of features names.

        Returns:
            model : Trained model.
        """

        if model == 'Logistic Regression':
            lr = LogisticRegression(random_state=79, max_iter=200)
            return lr.fit(self.X_train[features], self.y_train)
        elif model == 'Naive Bayes':
            gnb = GaussianNB()
            return gnb.fit(self.X_train[features], self.y_train)
        elif model =='Gradient Descent':
            sgd = SGDClassifier(loss='log_loss')
            return sgd.fit(self.X_train[features], self.y_train)
        elif model == 'Decision Tree':
            dt = DecisionTreeClassifier()
            return dt.fit(self.X_train[features], self.y_train)
        elif model == 'Adaboost':
            ada = AdaBoostClassifier(n_estimators=10, random_state=79)
            return ada.fit(self.X_train[features], self.y_train)
        elif model == 'Random Forest':
            rf = RandomForestClassifier(n_estimators=10, random_state=79)
            return rf.fit(self.X_train[features], self.y_train)
    
    def evaluate_logistic_regression(self):
        """ Train a Logistic Regression and compute a metric for each method using their chosen metric."""

        lr = LogisticRegression(random_state=79)

        # Getting the features for each method
        self.features['Logistic Regression']['Chi2 Filter'] = self.chi2_features
        self.features['Logistic Regression']['Variance Filter'] = self.var_features
        self.features['Logistic Regression']['Info Gain Filter'] = self.info_features
        self.features['Logistic Regression']['Forward Selection'] = self.ffs_wrapper(lr)
        self.features['Logistic Regression']['Backward Selection'] = self.bfs_wrapper(lr)
        self.features['Logistic Regression']['Lasso Importance'] = self.lasso_features
        self.features['Logistic Regression']['RF Importance'] = self.rf_features

        # Getting features for 'Chosen by One' e 'Chosen by All'
        # Formatting a dataframe with chosen features for each method 
        features = (
            pd.DataFrame(self.X_train.columns, columns=['Features'])
            .merge(pd.DataFrame(self.features['Logistic Regression']['Info Gain Filter'], columns=['Info Gain Filter']),
                   how='left', left_on='Features', right_on='Info Gain Filter')
            .merge(pd.DataFrame(self.features['Logistic Regression']['Forward Selection'], columns=['Forward Selection']),
                   how='left', left_on='Features', right_on='Forward Selection')
            .merge(pd.DataFrame(self.features['Logistic Regression']['Backward Selection'], columns=['Backward Selection']),
                   how='left', left_on='Features', right_on='Backward Selection')
            .merge(pd.DataFrame(self.features['Logistic Regression']['Lasso Importance'], columns=['Lasso Importance']),
                   how='left', left_on='Features', right_on='Lasso Importance')
            .merge(pd.DataFrame(self.features['Logistic Regression']['RF Importance'], columns=['RF Importance']),
                   how='left', left_on='Features', right_on='RF Importance')
        )
        chosen_by_one = []
        chosen_by_all = []
        for _, values in features.iterrows():
            if values.isnull().sum() == 0:
                chosen_by_one.append(values.fillna(values.mode()[0]).unique()[0])
                chosen_by_all.append(values.fillna(values.mode()[0]).unique()[0])
            elif values.isnull().sum() < 5:
                chosen_by_one.append(values.fillna(value=values.mode()[0]).unique()[0])
                chosen_by_all.append(np.nan)
        self.features['Logistic Regression']['Chosen by One'] = chosen_by_one
        try:
            if np.isnan(np.array(chosen_by_all)).all()==True:
                self.features['Logistic Regression']['Chosen by All'] = []
        except:
            self.features['Logistic Regression']['Chosen by All'] = [x for x in np.array(chosen_by_all) if x!='nan']

        # Getting the metrics for each method
        for m in self.methods:
            cols = self.features['Logistic Regression'][m]
            if len(cols) > 0:
                lr = self.train_model('Logistic Regression', cols)
                if self.metric_name in ['roc_auc', ' average_precision']:
                    preds = lr.predict_proba(self.X_test[cols])[:, 1]
                else:
                    preds = lr.predict(self.X_test[cols])
                self.metrics['Logistic Regression'][m] = self.metric(self.y_test, preds)
            else:
                self.metrics['Logistic Regression'][m] = np.nan
        print('Logistic Regression Done!')
            
    def evaluate_naive_bayes(self):
        """ Train a Naive Bayes and compute a metric for each method using their chosen metric."""

        gnb = GaussianNB()
        
        # Getting the features for each method
        self.features['Naive Bayes']['Chi2 Filter'] = self.chi2_features
        self.features['Naive Bayes']['Variance Filter'] = self.var_features
        self.features['Naive Bayes']['Info Gain Filter'] = self.info_features
        self.features['Naive Bayes']['Forward Selection'] = self.ffs_wrapper(gnb)
        self.features['Naive Bayes']['Backward Selection'] = self.bfs_wrapper(gnb)
        self.features['Naive Bayes']['Lasso Importance'] = self.lasso_features
        self.features['Naive Bayes']['RF Importance'] = self.rf_features

        # Getting features for 'Chosen by One' e 'Chosen by All'
        # Formatting a dataframe with chosen features for each method 
        features = (
            pd.DataFrame(self.X_train.columns, columns=['Features'])
            .merge(pd.DataFrame(self.features['Naive Bayes']['Info Gain Filter'], columns=['Info Gain Filter']),
                   how='left', left_on='Features', right_on='Info Gain Filter')
            .merge(pd.DataFrame(self.features['Naive Bayes']['Forward Selection'], columns=['Forward Selection']),
                   how='left', left_on='Features', right_on='Forward Selection')
            .merge(pd.DataFrame(self.features['Naive Bayes']['Backward Selection'], columns=['Backward Selection']),
                   how='left', left_on='Features', right_on='Backward Selection')
            .merge(pd.DataFrame(self.features['Naive Bayes']['Lasso Importance'], columns=['Lasso Importance']),
                   how='left', left_on='Features', right_on='Lasso Importance')
            .merge(pd.DataFrame(self.features['Naive Bayes']['RF Importance'], columns=['RF Importance']),
                   how='left', left_on='Features', right_on='RF Importance')
        )
        chosen_by_one = []
        chosen_by_all = []
        for _, values in features.iterrows():
            if values.isnull().sum() == 0:
                chosen_by_one.append(values.fillna(values.mode()[0]).unique()[0])
                chosen_by_all.append(values.fillna(values.mode()[0]).unique()[0])
            elif values.isnull().sum() < 5:
                chosen_by_one.append(values.fillna(value=values.mode()[0]).unique()[0])
                chosen_by_all.append(np.nan)
        self.features['Naive Bayes']['Chosen by One'] = chosen_by_one
        try:
            if np.isnan(np.array(chosen_by_all)).all()==True:
                self.features['Naive Bayes']['Chosen by All'] = []
        except:
            self.features['Naive Bayes']['Chosen by All'] = [x for x in np.array(chosen_by_all) if x!='nan']

        # Getting the metrics for each method
        for m in self.methods:
            cols = self.features['Naive Bayes'][m]
            if len(cols) > 0:
                lr = self.train_model('Naive Bayes', cols)
                if self.metric_name in ['roc_auc', ' average_precision']:
                    preds = lr.predict_proba(self.X_test[cols])[:, 1]
                else:
                    preds = lr.predict(self.X_test[cols])
                self.metrics['Naive Bayes'][m] = self.metric(self.y_test, preds)
            else:
                self.metrics['Naive Bayes'][m] = np.nan
        print('Naive Bayes Done!')
            
    def evaluate_gradient_descent(self):
        """ Train a Stochastic Gradient Descent and compute a metric for each method using their chosen metric."""

        sgd = SGDClassifier()

        # Getting the features for each method
        self.features['Gradient Descent']['Chi2 Filter'] = self.chi2_features
        self.features['Gradient Descent']['Variance Filter'] = self.var_features
        self.features['Gradient Descent']['Info Gain Filter'] = self.info_features
        self.features['Gradient Descent']['Forward Selection'] = self.ffs_wrapper(sgd)
        self.features['Gradient Descent']['Backward Selection'] = self.bfs_wrapper(sgd)
        self.features['Gradient Descent']['Lasso Importance'] = self.lasso_features
        self.features['Gradient Descent']['RF Importance'] = self.rf_features

        # Getting features for 'Chosen by One' e 'Chosen by All'
        # Formatting a dataframe with chosen features for each method
        features = (
            pd.DataFrame(self.X_train.columns, columns=['Features'])
            .merge(pd.DataFrame(self.features['Gradient Descent']['Info Gain Filter'], columns=['Info Gain Filter']),
                   how='left', left_on='Features', right_on='Info Gain Filter')
            .merge(pd.DataFrame(self.features['Gradient Descent']['Forward Selection'], columns=['Forward Selection']),
                   how='left', left_on='Features', right_on='Forward Selection')
            .merge(pd.DataFrame(self.features['Gradient Descent']['Backward Selection'], columns=['Backward Selection']),
                   how='left', left_on='Features', right_on='Backward Selection')
            .merge(pd.DataFrame(self.features['Gradient Descent']['Lasso Importance'], columns=['Lasso Importance']),
                   how='left', left_on='Features', right_on='Lasso Importance')
            .merge(pd.DataFrame(self.features['Gradient Descent']['RF Importance'], columns=['RF Importance']),
                   how='left', left_on='Features', right_on='RF Importance')
        )
        chosen_by_one = []
        chosen_by_all = []
        for _, values in features.iterrows():
            if values.isnull().sum() == 0:
                chosen_by_one.append(values.fillna(values.mode()[0]).unique()[0])
                chosen_by_all.append(values.fillna(values.mode()[0]).unique()[0])
            elif values.isnull().sum() < 5:
                chosen_by_one.append(values.fillna(value=values.mode()[0]).unique()[0])
                chosen_by_all.append(np.nan)
        self.features['Gradient Descent']['Chosen by One'] = chosen_by_one
        try:
            if np.isnan(np.array(chosen_by_all)).all()==True:
                self.features['Gradient Descent']['Chosen by All'] = []
        except:
            self.features['Gradient Descent']['Chosen by All'] = [x for x in np.array(chosen_by_all) if x!='nan']
        
        # Getting the metrics for each method
        for m in self.methods:
            cols = self.features['Gradient Descent'][m]
            if len(cols) > 0:
                lr = self.train_model('Gradient Descent', cols)
                if self.metric_name in ['roc_auc', ' average_precision']:
                    preds = lr.predict_proba(self.X_test[cols])[:, 1]
                else:
                    preds = lr.predict(self.X_test[cols])
                self.metrics['Gradient Descent'][m] = self.metric(self.y_test, preds)
            else:
                self.metrics['Gradient Descent'][m] = np.nan
        print('Gradient Descent Done!')
            
    def evaluate_decision_tree(self):
        """ Train a Decision Tree and compute a metric for each method using their chosen metric."""

        dt = DecisionTreeClassifier()

        # Getting the features for each method
        self.features['Decision Tree']['Chi2 Filter'] = self.chi2_features
        self.features['Decision Tree']['Variance Filter'] = self.var_features
        self.features['Decision Tree']['Info Gain Filter'] = self.info_features
        self.features['Decision Tree']['Forward Selection'] = self.ffs_wrapper(dt)
        self.features['Decision Tree']['Backward Selection'] = self.bfs_wrapper(dt)
        self.features['Decision Tree']['Lasso Importance'] = self.lasso_features
        self.features['Decision Tree']['RF Importance'] = self.rf_features

        # Getting features for 'Chosen by One' e 'Chosen by All'
        # Formatting a dataframe with chosen features for each method 
        features = (
            pd.DataFrame(self.X_train.columns, columns=['Features'])
            .merge(pd.DataFrame(self.features['Decision Tree']['Info Gain Filter'], columns=['Info Gain Filter']),
                   how='left', left_on='Features', right_on='Info Gain Filter')
            .merge(pd.DataFrame(self.features['Decision Tree']['Forward Selection'], columns=['Forward Selection']),
                   how='left', left_on='Features', right_on='Forward Selection')
            .merge(pd.DataFrame(self.features['Decision Tree']['Backward Selection'], columns=['Backward Selection']),
                   how='left', left_on='Features', right_on='Backward Selection')
            .merge(pd.DataFrame(self.features['Decision Tree']['Lasso Importance'], columns=['Lasso Importance']),
                   how='left', left_on='Features', right_on='Lasso Importance')
            .merge(pd.DataFrame(self.features['Decision Tree']['RF Importance'], columns=['RF Importance']),
                   how='left', left_on='Features', right_on='RF Importance')
        )
        chosen_by_one = []
        chosen_by_all = []
        for _, values in features.iterrows():
            if values.isnull().sum() == 0:
                chosen_by_one.append(values.fillna(values.mode()[0]).unique()[0])
                chosen_by_all.append(values.fillna(values.mode()[0]).unique()[0])
            elif values.isnull().sum() < 5:
                chosen_by_one.append(values.fillna(value=values.mode()[0]).unique()[0])
                chosen_by_all.append(np.nan)
        self.features['Decision Tree']['Chosen by One'] = chosen_by_one
        try:
            if np.isnan(np.array(chosen_by_all)).all()==True:
                self.features['Decision Tree']['Chosen by All'] = []
        except:
            self.features['Decision Tree']['Chosen by All'] = [x for x in np.array(chosen_by_all) if x!='nan']
        
        # Getting the metrics for each method
        for m in self.methods:
            cols = self.features['Decision Tree'][m]
            if len(cols) > 0:
                lr = self.train_model('Decision Tree', cols)
                if self.metric_name in ['roc_auc', ' average_precision']:
                    preds = lr.predict_proba(self.X_test[cols])[:, 1]
                else:
                    preds = lr.predict(self.X_test[cols])
                self.metrics['Decision Tree'][m] = self.metric(self.y_test, preds)
            else:
                self.metrics['Decision Tree'][m] = np.nan
        print('Decision Tree Done!')
            
    def evaluate_adaboost(self):
        """ Train a Adaboost and compute a metric for each method using their chosen metric."""

        ada = AdaBoostClassifier(n_estimators=10, random_state=79)

        # Getting the features for each method
        self.features['Adaboost']['Chi2 Filter'] = self.chi2_features
        self.features['Adaboost']['Variance Filter'] = self.var_features
        self.features['Adaboost']['Info Gain Filter'] = self.info_features
        self.features['Adaboost']['Forward Selection'] = self.ffs_wrapper(ada)
        self.features['Adaboost']['Backward Selection'] = self.bfs_wrapper(ada)
        self.features['Adaboost']['Lasso Importance'] = self.lasso_features
        self.features['Adaboost']['RF Importance'] = self.rf_features

        # Getting features for 'Chosen by One' e 'Chosen by All'
        # Formatting a dataframe with chosen features for each method 
        features = (
            pd.DataFrame(self.X_train.columns, columns=['Features'])
            .merge(pd.DataFrame(self.features['Adaboost']['Info Gain Filter'], columns=['Info Gain Filter']),
                   how='left', left_on='Features', right_on='Info Gain Filter')
            .merge(pd.DataFrame(self.features['Adaboost']['Forward Selection'], columns=['Forward Selection']),
                   how='left', left_on='Features', right_on='Forward Selection')
            .merge(pd.DataFrame(self.features['Adaboost']['Backward Selection'], columns=['Backward Selection']),
                   how='left', left_on='Features', right_on='Backward Selection')
            .merge(pd.DataFrame(self.features['Adaboost']['Lasso Importance'], columns=['Lasso Importance']),
                   how='left', left_on='Features', right_on='Lasso Importance')
            .merge(pd.DataFrame(self.features['Adaboost']['RF Importance'], columns=['RF Importance']),
                   how='left', left_on='Features', right_on='RF Importance')
        )
        chosen_by_one = []
        chosen_by_all = []
        for _, values in features.iterrows():
            if values.isnull().sum() == 0:
                chosen_by_one.append(values.fillna(values.mode()[0]).unique()[0])
                chosen_by_all.append(values.fillna(values.mode()[0]).unique()[0])
            elif values.isnull().sum() < 5:
                chosen_by_one.append(values.fillna(value=values.mode()[0]).unique()[0])
                chosen_by_all.append(np.nan)
        self.features['Adaboost']['Chosen by One'] = chosen_by_one
        try:
            if np.isnan(np.array(chosen_by_all)).all()==True:
                self.features['Adaboost']['Chosen by All'] = []
        except:
            self.features['Adaboost']['Chosen by All'] = [x for x in np.array(chosen_by_all) if x!='nan']
        
        # Getting the metrics for each method
        for m in self.methods:
            cols = self.features['Adaboost'][m]
            if len(cols) > 0:
                lr = self.train_model('Adaboost', cols)
                if self.metric_name in ['roc_auc', ' average_precision']:
                    preds = lr.predict_proba(self.X_test[cols])[:, 1]
                else:
                    preds = lr.predict(self.X_test[cols])
                self.metrics['Adaboost'][m] = self.metric(self.y_test, preds)
            else:
                self.metrics['Adaboost'][m] = np.nan
        print('Adaboost Done!')
            
    def evaluate_random_forest(self):
        """ Train a Random Forest and compute a metric for each method using their chosen metric."""

        rf = RandomForestClassifier(n_estimators=10, random_state=79)

        # Getting the features for each method
        self.features['Random Forest']['Chi2 Filter'] = self.chi2_features
        self.features['Random Forest']['Variance Filter'] = self.var_features
        self.features['Random Forest']['Info Gain Filter'] = self.info_features
        self.features['Random Forest']['Forward Selection'] = self.ffs_wrapper(rf)
        self.features['Random Forest']['Backward Selection'] = self.bfs_wrapper(rf)
        self.features['Random Forest']['Lasso Importance'] = self.lasso_features
        self.features['Random Forest']['RF Importance'] = self.rf_features

        # Getting features for 'Chosen by One' e 'Chosen by All'
        # Formatting a dataframe with chosen features for each method 
        features = (
            pd.DataFrame(self.X_train.columns, columns=['Features'])
            .merge(pd.DataFrame(self.features['Random Forest']['Info Gain Filter'], columns=['Info Gain Filter']),
                   how='left', left_on='Features', right_on='Info Gain Filter')
            .merge(pd.DataFrame(self.features['Random Forest']['Forward Selection'], columns=['Forward Selection']),
                   how='left', left_on='Features', right_on='Forward Selection')
            .merge(pd.DataFrame(self.features['Random Forest']['Backward Selection'], columns=['Backward Selection']),
                   how='left', left_on='Features', right_on='Backward Selection')
            .merge(pd.DataFrame(self.features['Random Forest']['Lasso Importance'], columns=['Lasso Importance']),
                   how='left', left_on='Features', right_on='Lasso Importance')
            .merge(pd.DataFrame(self.features['Random Forest']['RF Importance'], columns=['RF Importance']),
                   how='left', left_on='Features', right_on='RF Importance')
        )
        chosen_by_one = []
        chosen_by_all = []
        for _, values in features.iterrows():
            if values.isnull().sum() == 0:
                chosen_by_one.append(values.fillna(values.mode()[0]).unique()[0])
                chosen_by_all.append(values.fillna(values.mode()[0]).unique()[0])
            elif values.isnull().sum() < 5:
                chosen_by_one.append(values.fillna(value=values.mode()[0]).unique()[0])
                chosen_by_all.append(np.nan)
        self.features['Random Forest']['Chosen by One'] = chosen_by_one
        try:
            if np.isnan(np.array(chosen_by_all)).all()==True:
                self.features['Random Forest']['Chosen by All'] = []
        except:
            self.features['Random Forest']['Chosen by All'] = [x for x in np.array(chosen_by_all) if x!='nan']
        
        # Getting the metrics for each method
        for m in self.methods:
            cols = self.features['Random Forest'][m]
            if len(cols) > 0:
                lr = self.train_model('Random Forest', cols)
                if self.metric_name in ['roc_auc', ' average_precision']:
                    preds = lr.predict_proba(self.X_test[cols])[:, 1]
                else:
                    preds = lr.predict(self.X_test[cols])
                self.metrics['Random Forest'][m] = self.metric(self.y_test, preds)
            else:
                self.metrics['Random Forest'][m] = np.nan
        print('Random Forest Done!')
        
    def evaluate_models(self):
        """ Train all models and compute their metric"""

        # Run all filters
        self.chi2_filter()
        self.variance_filter()
        self.information_filter()
        self.lasso_embedded()
        self.rf_embedded()
        
        # Evaluate all models
        self.evaluate_logistic_regression()
        self.evaluate_naive_bayes()
        self.evaluate_gradient_descent()
        self.evaluate_decision_tree()
        self.evaluate_adaboost()
        self.evaluate_random_forest()

        # Format a report dataframe with model/method and their metric
        self.report = (
            pd.DataFrame(self.metrics['Logistic Regression'].values(), index=self.metrics['Logistic Regression'].keys(), columns=['Logistic Regression'])
            .merge(
                pd.DataFrame(self.metrics['Naive Bayes'].values(), index=self.metrics['Naive Bayes'].keys(), columns=['Naive Bayes']),
                left_index=True, right_index=True
            )
            .merge(
                pd.DataFrame(self.metrics['Gradient Descent'].values(), index=self.metrics['Gradient Descent'].keys(), columns=['Gradient Descent']),
                left_index=True, right_index=True
            )
            .merge(
                pd.DataFrame(self.metrics['Decision Tree'].values(), index=self.metrics['Decision Tree'].keys(), columns=['Decision Tree']),
                left_index=True, right_index=True
            )
            .merge(
                pd.DataFrame(self.metrics['Adaboost'].values(), index=self.metrics['Adaboost'].keys(), columns=['Adaboost']),
                left_index=True, right_index=True
            )
            .merge(
                pd.DataFrame(self.metrics['Random Forest'].values(), index=self.metrics['Random Forest'].keys(), columns=['Random Forest']),
                left_index=True, right_index=True
            )
        )

        # Get the best set of model/method that get the metric
        models = self.report
        self.best_model = ''
        self.best_method = ''
        self.best_metric = -1
        for model in models:
            for key, value in self.report[model].items():
                if value > self.best_metric:
                    self.best_model = model
                    self.best_method = key
                    self.best_metric = value

        # Format best_settings attribute
        self.best_settings = {
            'Model': self.best_model,
            'Method': self.best_method,
            'Metric': self.best_metric,
            'Features': self.features[self.best_model][self.best_method]
        }