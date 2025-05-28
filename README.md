# Housing Price Explanation Dashboard
In this file some goals of the project and features of the dashboard will be mentioned as well as a way to run the dashboard. This dashboard is designed for data-science students who struggle with knowing what features are contributing to the price of a house and how much these features contribute. A prediction model (XGBoost) was made and SHAP and LIME are used for the dashboard to give more insights in the price.

## Project Goals
The main goals for this project were the following:
- Provide intuitive and visual explanations of the price using SHAP and LIME
- Make a comparison between these two methods
- Allow users to interact and explore these features and their impact
- Collect user feedback and tune the dashboard according to the feedback

## Features
- SHAP Waterfall plot; a step-by-step breakdown of the price on a single prediction, so a local explainer
- SHAP Summary Plot; Global explainer for feature importance across the test samples
- LIME Local Explanation; Feature influence a single prediction
- Top 5 Feature Comparison; Highlights agreement and disagreement between SHAP and LIME

## How-To-Run
The file Dashboard_FinalVersion.py should be ran and at the bottom of the file the following line should be run in the terminal of the IDE:

streamlit run Dashboard_FinalVersion.py

The dashboard should launch in a localhost environment, if it isn't automatically there, it can be copied from the output in the browser and it should work.
