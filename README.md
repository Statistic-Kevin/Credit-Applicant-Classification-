# Credit Applicant Classification

<p align="center">
<img src="Images/7553752-abgelehntes-und-genehmigtes-kredit-oder-darlehensformular-mit-klemmbrett-und-anspruchsformular-darauf-vektor.jpg" width="1300"/>
</p>

![GitHub last commit](https://img.shields.io/github/last-commit/Statistic-Kevin/Credit-Applicant-Classification)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Statistic-Kevin/Credit-Applicant-Classification)
[![](https://tokei.rs/b1/github/Statistic-Kevin/Credit-Applicant-Classification?category=lines)](https://github.com/Statistic-Kevin/Credit-Applicant-Classification) 
![GitHub Repo stars](https://img.shields.io/github/stars/Statistic-Kevin/Credit-Applicant-Classification?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Statistic-Kevin/Credit-Applicant-Classification?style=social)

## :credit_card: Classifying credit applicants [[Presentation]](https://github.com/Statistic-Kevin/Credit-Applicant-Classification-/blob/main/Presentation.pdf)[[Report]](https://github.com/Statistic-Kevin/Credit-Applicant-Classification-/blob/main/Report.pdf)

The goal of this project is to set up a model to classify credit applicants according to their credit risk (good vs bad), based on a dataset of applicants in the German bank. 
The model could be used by banks to support their credit decisions. Since the goal of of this project is to produce accurate predictions, models are not chosen based on their interpretability or ability to conduct 
statistical inference with them. 

Tested ML models:
* Parametric (Elastic Net Logistic Regression Model)
* Non-parametric (Multivariate Adaptive Regression Splines)
* Tree- based Ensemble Methods ( Random Forest and Bayesian Additive Regression Trees)

More information on the criteria of the choice of the ML methods is found in chapter 3 (Modelling) of the report [here](https://github.com/Statistic-Kevin/Credit-Applicant-Classification-/blob/main/Report.pdf).

                                                                                                                                                Modelling Phases
<p align="left">
<img src="Images//Modelling_Phases.PNG" width="700"/>

                                                                                                                      ROC Curves for all selected models 

<p align="center">
<img src="Images//ROCcurves for all selected models.jpg" width="700"/>

                                                                                                                      Distributions of Predictions on Test Set by Actual Response
                                                              
<p align="right">
<img src="Images//Distributions of predictions on test set, by actual response.jpg" width="700"/>
