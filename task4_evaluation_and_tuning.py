"""Task 4 - Evaluation and Tuning."""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    confusion_matrix,roc_curve,roc_auc_score,precision_recall_curve,
    mean_absolute_error,mean_squared_error,r2_score)
from sklearn.dummy import DummyRegressor
from loan_data import make_loan_data

def main():
    df=make_loan_data()
    y=df["default"]
    X=df.drop(columns=["default","application_date"])
    X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.4,random_state=42,stratify=y)
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42,stratify=y_temp)
    num=["age","income","employment_years","credit_score","loan_amount","debt_to_income"]
    cat=["employment_type","education","purpose","grade"]
    prep=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])
    classifiers={
        "LogisticRegression":LogisticRegression(max_iter=2000,random_state=42),
        "DecisionTree":DecisionTreeClassifier(max_depth=5,random_state=42),
        "RandomForest":RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=-1)
    }
    fitted={}
    print("CLASSIFIER METRICS ON VALIDATION SET")
    for name,model in classifiers.items():
        pipe=Pipeline([("preprocess",prep),("model",model)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_val); prob=pipe.predict_proba(X_val)[:,1]
        fitted[name]=pipe
        print(f"\n{name}")
        print(f"Accuracy: {accuracy_score(y_val,pred):.3f}")
        print(f"Precision: {precision_score(y_val,pred,zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_val,pred,zero_division=0):.3f}")
        print(f"F1: {f1_score(y_val,pred,zero_division=0):.3f}")
        print("Confusion matrix:\n",confusion_matrix(y_val,pred))
    try:
        from xgboost import XGBClassifier
        
        prep_xgb=prep
        Xt=prep_xgb.fit_transform(X_train,y_train); Xv=prep_xgb.transform(X_val)
        scale_pos=(len(y_train)-y_train.sum())/max(y_train.sum(),1)
        xgb=XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.06,
                          subsample=.9,colsample_bytree=.9,eval_metric="logloss",
                          random_state=42,n_jobs=-1,scale_pos_weight=scale_pos)
        xgb.fit(Xt,y_train,verbose=False)
        fitted["XGBoost"]=(prep_xgb,xgb)
        xgb_prob=xgb.predict_proba(Xv)[:,1]; xgb_pred=(xgb_prob>=.5).astype(int)
        print("\nXGBoost")
        print(f"Accuracy: {accuracy_score(y_val,xgb_pred):.3f}")
        print(f"Precision: {precision_score(y_val,xgb_pred,zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_val,xgb_pred,zero_division=0):.3f}")
        print(f"F1: {f1_score(y_val,xgb_pred,zero_division=0):.3f}")
        print("Confusion matrix:\n",confusion_matrix(y_val,xgb_pred))
    except ImportError:
        xgb=None; print("\nXGBoost unavailable; install xgboost to run the XGBoost sections.")

    
    plt.figure(figsize=(8,6))
    roc_models=["LogisticRegression","RandomForest"]
    for name in roc_models:
        p=fitted[name].predict_proba(X_val)[:,1]
        fpr,tpr,_=roc_curve(y_val,p); auc=roc_auc_score(y_val,p)
        plt.plot(fpr,tpr,label=f"{name} AUC={auc:.3f}")
    if xgb is not None:
        fpr,tpr,_=roc_curve(y_val,xgb_prob); auc=roc_auc_score(y_val,xgb_prob)
        plt.plot(fpr,tpr,label=f"XGBoost AUC={auc:.3f}")
    else:
        p=fitted["DecisionTree"].predict_proba(X_val)[:,1]
        fpr,tpr,_=roc_curve(y_val,p); auc=roc_auc_score(y_val,p)
        plt.plot(fpr,tpr,label=f"DecisionTree AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"k--",label="Random baseline")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_curves_auc.png",dpi=150); plt.close()
    print("\nROC curves saved: roc_curves_auc.png")

    
    threshold_probs = xgb_prob if xgb is not None else fitted["RandomForest"].predict_proba(X_val)[:,1]
    precision,recall,thresholds=precision_recall_curve(y_val,threshold_probs)
    eligible=np.where(recall[:-1]>=.75)[0]
    if len(eligible):
        best_idx=eligible[np.argmax(precision[:-1][eligible])]
        threshold=float(thresholds[best_idx])
        pred_thr=(threshold_probs>=threshold).astype(int)
        print(f"Threshold for recall >= 0.75: {threshold:.3f}")
        print(f"Recall at threshold: {recall_score(y_val,pred_thr):.3f}")
        print(f"Precision at threshold: {precision_score(y_val,pred_thr,zero_division=0):.3f}")
    else:
        print("No validation threshold achieved recall >= 0.75.")

    
    rf_pipe=Pipeline([("preprocess",prep),("model",RandomForestClassifier(random_state=42,n_jobs=-1))])
    rf_grid={"model__n_estimators":[80,120],"model__max_depth":[None,8,14],
             "model__min_samples_leaf":[1,2]}
    rf_search=GridSearchCV(rf_pipe,rf_grid,cv=5,scoring="f1",n_jobs=-1)
    rf_search.fit(X_train,y_train)
    print("\nRandomForest GridSearchCV best parameters:",rf_search.best_params_)
    print(f"RandomForest best CV F1: {rf_search.best_score_:.3f}")

    if xgb is not None:
        xgb_search=RandomizedSearchCV(
            XGBClassifier(eval_metric="logloss",random_state=42,n_jobs=-1,scale_pos_weight=scale_pos),
            {"n_estimators":randint(60,181),"max_depth":randint(2,7),
             "learning_rate":uniform(.02,.18),"subsample":[.75,.9,1.0]},
            n_iter=15,cv=5,scoring="f1",random_state=42,n_jobs=-1
        )
        xgb_search.fit(Xt,y_train)
        print("\nXGBoost RandomizedSearchCV best parameters:",xgb_search.best_params_)
        print(f"XGBoost best CV F1: {xgb_search.best_score_:.3f}")
        best_name,best_est=("XGBoost",xgb_search.best_estimator_) if xgb_search.best_score_>=rf_search.best_score_ else ("RandomForest",rf_search.best_estimator_)
    else:
        best_name,best_est="RandomForest",rf_search.best_estimator_

    
    reg_cols=["age","income","employment_years","credit_score","debt_to_income"]
    Xr=df[reg_cols].copy(); yr=df["loan_amount"]
    Xr_train,Xr_test,yr_train,yr_test=train_test_split(Xr,yr,test_size=.2,random_state=42)
    med=Xr_train.median(); Xr_train=Xr_train.fillna(med); Xr_test=Xr_test.fillna(med)
    reg=LinearRegression().fit(Xr_train,yr_train)
    base=DummyRegressor(strategy="mean").fit(Xr_train,yr_train)
    for name,model in [("LinearRegression",reg),("Mean baseline",base)]:
        pr=model.predict(Xr_test)
        print(f"\nRegression {name}: MAE {mean_absolute_error(yr_test,pr):.2f} | RMSE {np.sqrt(mean_squared_error(yr_test,pr)):.2f} | R2 {r2_score(yr_test,pr):.3f}")

    
    X_test_t=prep_xgb.transform(X_test) if xgb is not None else None
    if best_name=="XGBoost" and xgb is not None:
        test_pred=best_est.predict(X_test_t); test_prob=best_est.predict_proba(X_test_t)[:,1]
    else:
        test_pred=best_est.predict(X_test); test_prob=best_est.predict_proba(X_test)[:,1]
    print(f"\nBest model: {best_name} | Test ROC-AUC {roc_auc_score(y_test,test_prob):.3f} | Test F1 {f1_score(y_test,test_pred):.3f}")
    print("Accuracy can mislead on imbalanced data because a model can predict the majority class most of the time and still achieve high accuracy while missing many minority-class cases.")
if __name__=="__main__": main()
