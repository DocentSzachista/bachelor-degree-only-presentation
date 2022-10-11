import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from collections import defaultdict

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay


def my_confusion_matrixs(
    data, 
    ax,
    labels : list[str], 
    colors : list[str],
    title : str | None ="", 
    colors_description : str | None="Values intensity",
    is_corr = False
    ):
    # fig, ax = plt.subplots()
    img = ax.imshow(data, cmap=colors[1])
    cbar = ax.figure.colorbar(img, ax=ax)
    # cbar.ax.set_ylabel(colors_description, rotation=-90, va="bottom")
    ax.set_xticks( np.arange(len(labels)), labels=labels, rotation=45)
    ax.set_xlabel("predicted values")
    ax.set_ylabel("actual values")
    ax.set_yticks( np.arange(len(labels)), labels=labels)
    ax.set_title(title)
    if not is_corr :
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, data[i, j],
                    ha="center", va="center", color=colors[0] 
                )
    ax.plot()


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def find_best_parameters_estimator(
    x_train_data, 
    y_train_data, 
    classifier,  
    resampler,
    classifier_parameters,
    cv = 5
    ):
    steps = [
        ("sampler", resampler),
        ("cls", classifier)
    ]
    score = {
        'F1':'f1'
    }
    pipe = Pipeline(steps=steps)
    grid_search_CV = GridSearchCV(
        estimator=pipe, 
        param_grid=classifier_parameters, 
        scoring= score, cv=cv, refit="F1",  
        )


    grid_search_CV.fit(x_train_data.values, y_train_data.values)
    
    grid_score = {
            "names": f"{type(classifier).__name__}-{type(resampler).__name__}", 
            "scores" : grid_search_CV.best_score_,
            "estimator": grid_search_CV.best_estimator_,
            "best_params": grid_search_CV.best_params_ 
        }
    return grid_score 


def find_best_parameters_estimator_no_preprocessing(
    x_train_data, 
    y_train_data, 
    classifier,  
    classifier_parameters,
    cv = 5
):
    grid_search_CV = GridSearchCV(
        estimator=classifier, 
        param_grid=classifier_parameters, 
        scoring= ["f1", "balanced_accuracy"], cv=cv, refit="balanced_accuracy", n_jobs=-1 )

    grid_search_CV.fit(x_train_data.values, y_train_data.values)
    
    grid_score = {
            "scores" : grid_search_CV.best_score_,
            "estimator": grid_search_CV.best_estimator_,
            "best_params": grid_search_CV.best_params_ 
        }
    return grid_score 

def train_predict_and_count_confusion_matrix(
    x_train, y_train, x_test, y_test, classifier):
    
    classifier.fit(x_train.values, y_train.values)
    y_predictions = classifier.predict(x_test.values)
    conf_matrix = confusion_matrix(y_test, y_predictions)
    return conf_matrix 


def calibrate_classifier(classifier, x_train, x_test, y_train, y_test, method="isotonic", cv=5):
    """### Calibrates the classifier with used method
    
    Description:
    ------------

    Computes propability callibration on provided classifier
    and later displays:

    - Plot with the average predicted probability 
      for each bin on the x-axis and the fraction of positive classes 
      in each bin on the y-axis.
                    
    - Histogram plots showing mean predicted probability
                    
    - precision, recall, f1 and auc-roc scores to indicate how 
                    much predictions have been made better  

    Returns 
    -------
    - confusion matrixes showin physically how model has improved
    """
    # Setup plots
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4,2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    to_plot_list = [
        (type(classifier).__name__, classifier),
        (
            f"{type(classifier).__name__} calibrated ",
            CalibratedClassifierCV(classifier, cv=cv, method=method)
        )
    ]
    displays = {}
    matrixes = []
    grid_pos = [(2, 0), (2, 1)]
    # display on one plot probs before and after calibration
    for i, (name, clf) in enumerate(to_plot_list) :
        clf.fit(x_train, y_train)
        display = CalibrationDisplay.from_estimator(
            clf,
            x_test,
            y_test,
            n_bins=10,
            name= name,
            ax=ax_calibration_curve,
        )
        displays[name] = display 
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")
    
    # Display histograms of classificators before and after calibration
    for i, (name, _) in enumerate(to_plot_list):
        row, col = grid_pos[i]
       
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            # color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    scores = defaultdict(list)
    matrixes = []

    for i, (name, clf) in enumerate(to_plot_list):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        matrixes.append((name,confusion_matrix(y_test, y_pred)))
        scores["Classifier"].append(name)
        for metric in [precision_score, recall_score, f1_score, roc_auc_score ]:
            score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            scores[score_name].append(metric(y_test, y_pred))
    
    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df.round(decimals=3)    
   
    plt.tight_layout()
    plt.show()
    
    print(score_df)
    return matrixes

def count_scores(classifier, x_test, y_test, scores, com = ""):
    y_pred = classifier.predict(x_test)
    scores["cls"].append(f"{type(classifier).__name__} {com}")
    for metric in [precision_score, recall_score, f1_score, roc_auc_score]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_pred))

    


def roc_plot (classifier, ax, test_x, test_y, title  ):
    """
        ### Draw ROC curve for provided classifier on given axes

        ### Params:
            - classifier: Your classifier that you want to check
            - ax: axis where plot is being drawn
            - test_x: test data from dataset 
            - test_y: dataset output
            - title: title for curve 
    """
    ns_probs = [0 for _ in range(len(test_y))]
    probs = classifier.predict_proba(test_x)[:, 1]
    ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
    fpr, tpr, _ = roc_curve(test_y, probs)
    ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    ax.plot(
        fpr, tpr, marker='.', label=f"{type(classifier).__name__}"
    )
   

    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # show the legend
    ax.legend()
   
def plot_matrixes(matrix_list, matrix_class_desc): 
    fig, ax = plt.subplots(1, len(matrix_list), figsize= (12, 8))
    for i in range(len(matrix_list)):
        my_confusion_matrixs(
            matrix_list[i]['matrix'],
            ax[i], 
            labels = matrix_class_desc, 
            colors = ["red", "plasma"],
            title = matrix_list[i]["name"]
            )
            
    plt.subplots_adjust(
        left=0.0,
        bottom=0.2, 
        right=1.5, 
        top=.8, 
        wspace=0.3, 
        hspace=0.4
    )