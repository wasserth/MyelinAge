import os
import numpy as np
import pandas as pd
from p_tqdm import p_map
# import scipy
# import scipy.stats
import scipy.stats as st
import pingouin as pg
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.linear_model import LinearRegression


"""
Significance tests info: 

Wilcoxon:   - non-parametric 
            - paired data
            - for ordered data (not for nominal data)
            -> used for Neuroimage paper (on Dice score per subject to compare methods)
            (scipy.stats.wilcoxon)
            (non-parametric version of ttest_rel)

McNemar:    - non-parametric 
            - paired data
            - for nominal data (=categorical data/classification results) 
            -> good for classification predictions (because those are categorical)
            -> only works for 2 classes (binary)
            -> used in master thesis 
            -> is the special case of "chi-squared symmetry test" for binary data
            -> good explanation for classifiers: https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
            -> does not say which classifier is better, only if the error distribution is similar => not useful for me ?

- Depending on the problem those tests might not be meaningful on the raw predictions, but on the metric/Difference (as input)
  (e.g. Dice, error): Because only the predictions have a relation to groundtruth and therefore we could only say
  if two models' predictions are different but not if one of them is better in relation to the groundtruth.
- Nullhypothese: group a and b are identical. If p<0.05 the nullhypothesis is wrong and the groups are different.
- Building complex linear models (e.g. with iteration effects):
    https://github.com/eigenfoo/tests-as-linear/blob/master/tests-as-linear.ipynb
    import statsmodels.formula.api as smf
    smf.ols('age ~ 1 + pred_m', data=combined).fit().summary()
"""

def mcnemar_test(gt, pred_a, pred_b):
    """
    Make McNemar test for predictions of 2 classifiers

    gt: 1d array    (groundtruth)
    pred_a: 1d array
    pred_b: 1d array
    """
    # Define contingency table (https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)
    table = [[0, 0],
             [0, 0]]
    for idx, elem in enumerate(gt):
        if pred_a[idx] == gt[idx]:
            if pred_b[idx] == gt[idx]:
                table[0][0] += 1  # both predictions correct
            else:
                table[0][1] += 1  # pred_a correct, pred_b incorrect
        else:
            if pred_b[idx] == gt[idx]:
                table[1][0] += 1  # pred_a incorrect, pred_b correct
            else:
                table[1][1] += 1  # both predictions incorrect
    mcnemar_pval = mcnemar(table, exact=True).pvalue
    return mcnemar_pval
    

def bootstrap_statistics(data,
                         statfunction=lambda x: np.average(x),
                         null_hypo_thr=0,
                         alpha=0.05,
                         n_samples=10000,
                         n_cores=2):
    """
    Calc confidence interval and pval with bootstrapping.

    statfunction: input: data subset, output: a single float score
                    (can simply be the mean over errors)
                    (can calc accuracy of two classifiers vs groundtruth and take the difference. In this case,
                    the better classifier should be the first one (mean of difference should be positive))
    null_hypo_thr: the treshold of the null hypothesis to get a p_value
                   null hypothesis: output by statfunction is <= null_hypo_thr
                    (for comparing two classifiers it should be 0, because null hypothesis is that difference is 0)
                    (for comparing accuracy of a single binary classifier to random, should be 0.5, because random would be 0.5)
    alpha: the alpha to get CI for 
    n_samples: nr of bootstrapping iterations 
    n_cores: nr of threads to use

    returns: CI 2.5%, CI 97.5%, pval
    """
    data = np.array(data) # cast from e.g. pd.Series to np.array -> leads to great speedup!!

    def bootstrap_iteration(idx):
        # select random subset of dataset (elements can repeat to make subset have same size as original dataset)
        rand_idxs = np.random.randint(data.shape[0], size=(data.shape[0]))
        rand_subset = data[rand_idxs]
        return statfunction(rand_subset) # apply function of interest to the subset

    if n_cores == 1:
        stats_subsets = np.array([bootstrap_iteration(idx) for idx in range(n_samples)])
    else:
        # Does not scale well above 2-4 cores  (2x speedup over single core)
        # If fast statsfunction (e.g. average) this is >10x slower than singlecore  (same if using Parallel)
        stats_subsets = np.array(p_map(bootstrap_iteration, range(n_samples), num_cpus=n_cores, disable=True))

    # For two algorithms (e.g. want to show that B is better than A):
    # null-hypothesis: B is equal or better than A. So the difference in accuracy scores must be <= 0.
    # pval: count how often is the difference <=0 (how often B is really better than A)
    #  -> if is less than 0.05, then only in 5% of cases B was really better
    pval = (stats_subsets <= null_hypo_thr).sum() / n_samples

    alpha = alpha / 2  # divide by 2 because left and right together should be 5%
    return np.quantile(stats_subsets, alpha).round(5), np.quantile(stats_subsets, 1-alpha).round(5), pval


def ci_bootstrap(data, statfunction=np.average, alpha=0.05, n_samples=10000):
    """
    Calculate the confidence interval using non-parametric bootstrapping.
    Default is 95% CI.

    Confidence intervals: 
    Which interval the true population mean is in. Only works if we can assign one score to each subject (e.g. 
    one Dice per subject. For classification however F1 is calculated via sum of FP/FN/TP and therefore there is
    no F1 per single subject. Works for regression: one MAE per subject)
    
    Could also work for classification: Permute raw results and then after each permutation calc the F1. Then in 
    the end I have a list of F1 score which I can take the percentiles from.

    (CI are not used to compare experiments).


    Very similar results to:
    import scikits.bootstrap as bootstrap    # small random package -> not ideal
    bootstrap.ci(data=my_data, statfunction=scipy.mean)

    https://stackoverflow.com/questions/44392978/compute-a-confidence-interval-from-sample-data-assuming-unknown-distribution/66008548#66008548
    inspired by https://github.com/cgevans/scikits-bootstrap
    """
    import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))    
    
    alphas = np.array([alpha/2, 1 - alpha/2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples-10):
        warnings.warn("Some values used extremal samples; results are probably unstable. "
                      "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()
    
    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals]

    
def ci_parametric(data, alpha=0.05):
    """
    Calculate the confidence interval using a parametric model (normal distribution).
    Default is 95% CI.

    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    https://www.statology.org/confidence-intervals-python/
    """
    alpha = 1-alpha
    # ci = st.t.interval(alpha, len(data)-1, loc=np.mean(data), scale=st.sem(data))  # for n<30
    ci = st.norm.interval(alpha, loc=np.mean(data), scale=st.sem(data))  # for n>30 (assuming a normal distribution)
    return ci


def ICC(ratings):
    """
    Calculate the Intraclass correlation coefficient.

    ratings: List of ratings from different raters of the same subjects.

    Not sure if this interpretation of pval is right. Probably not!!:
    If pval>0.05: the raters are significantly different
    If pval<0.05: the raters agree
    (this is inverse from normal significance test where p<0.05 means that groups are different)

    Notes about different measures for interrater agreement:
        ICC:            - allows several raters
                        - for continuous data
                        (does not behave well if ratings are only 0 and 1)
                        (ist quasi wie eine normale Korrelation)
        Cohen kappa:    - only for 2 raters
                        - for nominal data (classification results) (with quadratic weighting can be adapted for ordinal data?)
                        (from sklearn.metrics import cohen_kappa_score)
        Fleiss Kappa:   - for more than 2 raters
                        - for nominal data
                        (nr of raters per category should be roughly balanced to work properly)
                        (from statsmodels.stats.inter_rater import fleiss_kappa)
    """
    data_icc = []
    for idx, rating in enumerate(ratings):
        subject_ids = list(range(len(rating)))  # unique ID for each sample
        rater = np.ones(len(rating)) * idx  # unique ID for each rater
        rating = rating
        data_icc.append(np.stack([subject_ids, rater, rating], axis=1))
    data_icc = np.concatenate(data_icc)
    df_icc = pd.DataFrame(columns=["subject_id", "rater", "rating"], data=data_icc)
    icc = pg.intraclass_corr(data=df_icc, targets='subject_id', raters='rater', ratings='rating')
    icc = icc.round(5)
    # Select ICC for case where each rater rates each target  
    # (e.g. ICC1 would be that each sample is rated by a different rater)
    icc = icc.query("Type == 'ICC3'").iloc[0]
    return icc


def unconfound(y, confound, group_data=False):
    """
    This will remove the influence "confound" has on "y".

    If the data is made up of two groups, the group label (indicating the group) must be the first column of
    'confound'. The group label will be considered when fitting the linear model, but will not be considered when
    calculating the residuals.

    Args:
        y: [samples, targets]
        confound: [samples, confounds]
        group_data: if the data is made up of two groups (e.g. for t-test) or is just
                    one group (e.g. for correlation analysis)
    Returns:
        y_correct: [samples, targets]
    """
    # Demeaning beforehand or using intercept=True has similar effect
    #y = demean(y)
    #confound = demean(confound)

    lr = LinearRegression(fit_intercept=True).fit(confound, y)  # lr.coef_: [targets, confounds]
    if group_data:
        y_predicted_by_confound = lr.coef_[:, 1:] @ confound[:, 1:].T
    else:
        y_predicted_by_confound = lr.coef_ @ confound.T  # [targets, samples]
    y_corrected = y.T - y_predicted_by_confound
    return y_corrected.T  # [samples, targets]
