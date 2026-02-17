from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChurnModels:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.log_reg = None
        self.dt = None
        self.feature_names = None
    
    def train_models(self, feature_names):
        self.feature_names = feature_names
        
        print("training logistic regression")
        self.log_reg = LogisticRegression(random_state=42, max_iter=1000)
        self.log_reg.fit(self.X_train, self.y_train)
        
        print("training decision tree")
        self.dt = DecisionTreeClassifier(random_state=42, max_depth=10)
        self.dt.fit(self.X_train, self.y_train)
        
        print("models trained")
    
    def predict_proba(self, X):
        log_proba = self.log_reg.predict_proba(X)[:, 1]
        dt_proba = self.dt.predict_proba(X)[:, 1]
        return log_proba, dt_proba
    
    def predict(self, X):
        log_pred = self.log_reg.predict(X)
        dt_pred = self.dt.predict(X)
        return log_pred, dt_pred
    
    def get_metrics(self, y_true, log_pred, dt_pred):
        metrics = {}
        for name, pred in [('Logistic Regression', log_pred), ('Decision Tree', dt_pred)]:
            metrics[name] = {
                'Accuracy Score': accuracy_score(y_true, pred),
                'Precision Score': precision_score(y_true, pred),
                'Recall Score': recall_score(y_true, pred),
                'F1 Score': f1_score(y_true, pred)
            }
        return metrics
    
    def get_insights(self):
        dt_importance = self.dt.feature_importances_
        lr_coef = self.log_reg.coef_[0]
        return dt_importance, lr_coef
