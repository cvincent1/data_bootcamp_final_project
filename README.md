# Problem Statement:
Wine quality is a metric in the winemaking industry that can 'make or break' the success seen by a vineyard. Reviews and scoring are used in shaping consumer preferences, guiding production standards, and determining market success. For my final project, I wanted to develop a predictive model capable of accurately estimating the quality of red wines. I come from an Italian/English family that loves red wine, so when I was parsing through datasets on Kaggle, this caught my eye. The dataset I found focuses specifically on Portugal's renowned "Vinho Verde" variety. Vinho Verde, a traditional wine from the northern region of Portugal, comes in both red and white wine. While the white variant is more widely recognized for its light, crisp taste, the red Vinho Verde is equally unique, offering bold flavors and a distinct tartness. Both wines are renowned for their versatility and refreshing qualities. I have actually had the pleasure of trying Vinho Verde (Professor Koehler, I am 21 years old :) btw) when I noticed it was available at Astor Wines. I had the 2022 Vinho Verde "Biotite" by António Lopes Ribeiro, and it was quite good!

This project will focus specifically on the red variant of Vinho Verde, utilizing a dataset of 'physicochemical' properties and sensory outputs to predict wine quality scores on a scale from 0 to 10(although the scores range in the dataset from 3 to 8). These scores, typically determined through analysis by wine experts, reflect the wine's success with consumers. Developing a predictive model like this could have the potential to provide valuable insights into how specific chemical properties, such as acidity, sugar content, pH, and alcohol, can impact the perceived quality of a bottle of wine. Producers, winemakers, or vineyard owners can use these insights to make changes to their production methods, ensure better consistency, and create wines that align more closely with consumer expectations based on the model's findings.

# Dataset Description:

The dataset used for this project is sourced from Kaggle and contains detailed information on the properties of red Vinho Verde wines. This dataset includes features such as fixed acidity, volatile acidity, citric acid, residual sugar, pH, sulfur dioxide levels, alcohol content, and other variables that influence the overall profile of red wines. Additionally, it contains a wine quality score assigned by trained tasters, which serves as the target variable for the regression model. While this data provides a great amount of information, challenges may arise due to potential outliers in the dataset, as well as the naturally subjective nature of wine quality scoring. Before diving into testing, I will take a few steps to clean the data and address any missing values or anomalies that may come up.

# Lets create some graphs to visualize the data!
Similar to a test I performed in my midterm project, I will look to identify which numerical features have the strongest correlation with quality in the wine dataset and determine the statistical significance of these relationships. I will do this by calculating Pearson correlation coefficients and p-values for all numerical features against the target quality. Then, I will sort them into a DataFrame to rank features based on their correlation strengths.

The resulting table displays the following for each feature: 

Correlation Coefficient:

Positive Correlations:
Alcohol (0.476): This has the strongest positive relationship with quality. Higher alcohol content appears to correlate with better wine quality.
Sulfates (0.251): Positively correlated with quality as well, though weaker than Alcohol's correlation.

Negative Correlations:
Volatile acidity (-0.390): This test shows a strong negative correlation, meaning higher volatile acidity (essentially a vinegary taste) tends to decrease quality.

Density, Total sulfur dioxide, and Chlorides are also negatively correlated with quality but have relatively low correlations. ps (P-Value):

P-values close to 0, such as 2.831477e-91 for Alcohol, indicate strong statistical significance. The smaller the p-value, the stronger the evidence that the correlation is not due to random chance.

Residual sugar has a very high p-value (0.583), indicating no significant relationship with wine quality. As a result, I will not consider residual sugar moving forward, as it has little effect on perceived wine quality.


I will now move into visualizing the relationships between wine quality and some variables that had higher levels of correlation with it.

This graph shows the relationship between volatile acidity and wine quality scores. As the quality score increases, volatile acidity decreases significantly, suggesting an inverse relationship between the two. Wines with lower volatile acidity tend to receive higher quality ratings. Based on a quick Google search, I learned that this trend is common across most wines. High volatile acidity, which indicates higher levels of acetic acid (vinegar-like flavors), is typically considered undesirable and negatively affects wine quality. Oenological studies support this inverse relationship and is a consistent finding in wine quality research. I figured that a line graph was appropriate here because it effectively shows the trend or pattern in the relationship between the volatile acidity and quality score more than a bar graph would.

This bar graph shows the relationship between citric acid concentration and quality scores of wine. As the wine quality score increases, the average citric acid concentration also rises, suggesting a positive relationship between citric acid levels and wine quality. Higher quality wines seem to have higher citric acid content. Again, this seems to represent wine generally speaking as citric acid contributes to the freshness and acidity balance of wines, which is often associated with a more pleasant and crisp taste. Wines with higher citric acid levels are typically viewed as having better structure and complexity, leading to higher quality ratings.

This line graph shows the relationship between alcohol concentration and quality scores of wine. As the wine quality score increases, the alcohol concentration also rises, indicating a strong positive correlation. Higher quality wines tend to have higher alcohol levels. Higher alcohol content often results from riper grapes, which produce more sugar for fermentation, leading to richer and more full-bodied wines. Wines with higher alcohol levels are frequently associated with better structure, balance, and perceived quality.

This violin plot visualizes the distribution of sulphates concentration across different wine quality scores. As wine quality increases, the median sulphates concentration also tends to rise slightly, with the distributions becoming tighter for higher quality scores. Higher quality wines tend to have sulphates values clustered around a slightly elevated range. As it turns out, sulphates contribute to wine stability and help preserve freshness and aroma, which can positively influence quality. Elevated sulphates, when well-balanced, are associated with higher-quality wines, particularly in controlled production processes. I liked the idea of using violin plots here because it provided a more detailed view than something like a box plot by visualizing the density of the data, highlighting areas where most sulphates values occur.

# Multiple Regression Model
Moving forward with a multiple regression model is ideal for testing multiple variables against the wine quality as it can evaluate the relationship between several wine properties and the quality score at the same time. The model’s coefficients will likley provide clearer insights into how each variable impacts quality, such as alcohol positively contributing and volatile acidity negatively affecting it. A multiple regression model will also identifies the most significant predictors for me, while controlling for the influence of other features. This data can hypothetically help consumers and wine producers prioritize factors that matter most.

The results from my Multiple Linear Regression Model showed that alcohol was the strongest predictor of wine quality, followed by volatile acidity and sulphates. Features like density, residual sugar, and citric acid contributed minimally. The model achieved a MSE of 0.385 on the training data and 0.404 on the test data.
When compared to the baseline model, which had an MSE of 0.6517, the regression model significantly reduced prediction error, showing that incorporating the key features identified in the data meaningfully improves predictive accuracy. These results further validated how important properties like alcohol and acidity can be in determining wine quality and highlight the regression model’s effectiveness over the simpler baseline approach.
    
# K-Nearest Neighbors (KNN) Regression Model

A K-Nearest Neighbors (KNN) model should be well-suited for testing multiple variables against wine quality because it captures non-linear relationships and adapts to patterns within the data. By evaluating similarities between wines based on all variables, I hope that my KNN model can effectively integrate multiple predictors without assuming a specific relationship between them. Additionally,optimizing n_neighbors ensures the model balances bias and variance, further improving its accuracy.

Interestingly, after performing my K-Nearest Neighbors Test, the model achieved a training MSE of 0.3701 and a test MSE of 0.4454, slightly underperforming compared to the Linear Regression model (training MSE = 0.385, test MSE = 0.404). From the feature importance analysis table, columns like density, volatile acidity, and sulfates were identified as the most influential predictors, differing from Linear Regression, where alcohol was the dominant predictive variable. This discrepancy may be due to a KNN test's focus on local relationships within the data, capturing non-linear patterns that would end up emphasizing features like density. In wine, density is closely tied to other important attributes, such as sugar content, alcohol concentration, and fermentation levels. Wines with similar densities might have clustered together due to shared characteristics in these properties, which KNN naturally identified when forming local neighborhoods. Further, according to articles online, KNN might not be suitable for large datasets due to its computational complexity. The results from this test show that model choice can have an interesting effect on how feature importance is interpreted, depending on the data’s structure and relationships.

# Decision Tree Regression Model

Moving on, I would like to perform a Decision Tree Regression Model. A Decision Tree Regression Model is effective for testing multiple variables against wine quality because it can capture non-linear relationships and interactions between features, such as alcohol and sulphates. It ranks features based on their importance accross the entire dataset, providing clear insights into which variables most influence wine quality. Additionally, I like that decision trees are highly visual, showing exactly how splits are made during the process

The Decision Tree Regression Model identified alcohol as the most important feature, followed by sulphates and volatile acidity, while all other features had little to no importance. This lines up with my results from my Multiple Regression Model. The model’s optimal depth was determined as 3 through GridSearchCV. The MSE on the test dataset was 0.439, and the training set was 0.4299, showing reasonable fit but slightly higher error compared to the KNN and Linear Regression models. The tree visualization highlighted alcohol as the primary splitting variable, reaffirming its strong predictive relationship with wine quality. Unlike KNN, the Decision Tree did not emphasize features like density, suggesting it prioritizes variables with imapact accross the entire dataset. These results demonstrate the Decision Tree’s strength in identifying dominant features but also its limitations in capturing subtle patterns compared to models like KNN.

# Random Forest Regression Model

I have gone with a A Random Forest Regression Model for my final test. Random Forest Regression Model's are ideal for testing multiple variables against wine quality because it combines the strengths of multiple decision trees to improve its accuracy and reduce overfitting. By averaging predictions from numerous trees, it will be able to capture non-linear relationships and interactions between the features in my dataset. Additionally, these models handle large numbers of variables effectively and reduce bias by using randomized splits, making them well-suited for complex datasets like my own.

The Random Forest Regression Model achieved strong performance, with a test MSE of 0.349, and a train MSE of 0.049, the lowest among all tested models. Hyperparameter tuning identified 150 estimators and a maximum depth of 30 as the optimal settings, improving the model’s ability to generalize. Interestingly, sulphates and density ranked as the most important features, followed by volatile acidity, while alcohol, previously dominant in Linear Regression and Decision Tree models, ranked much lower. This difference arises because Random Forests reduce overfitting by averaging predictions across multiple decision trees, where other variables with local importance may dilute splits based on alcohol. Additionally, sulfates and density may have stronger non-linear relationships with wine quality, which Random Forests are better at detecting compared to linear or single-tree models.

# Key Findings:

Random Forest Regression Performance: The Random Forest Regression Model emerged as the best-performing model, achieving the lowest MSE of 0.349 on the test data, compared to 0.404 for Multiple Linear Regression, 0.445 for KNN, and 0.439 for Decision Tree. Most Influential Features: Across all the models, we saw some variation in the most influential features, with alcohol, density, sulphates, and volatile acidity being featured as the most important predictors of wine quality in different tests.

Model-Specific Insights:

Linear Regression: Alcohol was the most significant predictor due to its strong linear relationship with wine quality.

KNN: Density gained importance as KNN captured non-linear relationships within the data.

Decision Tree: Alcohol remained the top predictor, while the tree structure provided interpretability at the cost of slightly higher MSE.

Random Forest: In the best performing model overall, sulphates and density emerged as the most important features, as Random Forest effectively captured non-linear relationships and feature interactions by averaging predictions across multiple decision trees.

Feature Interactions: The Random Forest model’s results suggest that features like density and sulphates have subtle but crucial interactions with other variables that linear models were simply unable to capture.

Baseline Comparison: All models outperformed the baseline MSE of 0.6517, demonstrating their ability to predict wine quality and emphasizing the importance of different wine properties in influencing wine ratings.

# Next Steps & Potential Improvements:

Incorporate Additional Features:

I think that introducing external factors like grape variety, region-specific climate data, or production techniques has the potential to improve model performance and make for interesting results. Including aspects like sensory descriptions (taste profiles and aroma notes) to provide more context for wine quality scores could also be beneficial.

More Advanced Feature Engineering:

I think that transforming or combining existing features within the dataset, such as creating an alcohol-to-density ratio, to uncover hidden relationships could be interesting in gaining a better understaning of how certain variables in wine interact iwth each other

Cross-Validation and Scaling:

Using some additional cross-validation techniques to ensure robustness and consistency of model results. Doing this could improve models like the K-Nearest Neighbors Regression Model.
