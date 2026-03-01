# DSI-Cohort8-ML-2

Team project on ML

This repository covers team project for DSI Cohort 8 ML Project Team ML2.
The aim of the project is to create a model that can predict if a Customer will sign up for a Term load based on their input dataset collection of demographic data, financial data and previous campaign details. This project also aims to export the probability of the same.

```
├── data
├──── processed
├──── raw
├──── sql
├── experiments
├── models
├── reports
├── src
├── README.md
└── .gitignore
```

Data: Contains the raw, processed and final data. For any data living in a database, make sure to export the tables out into the sql folder, so it can be used by anyone else.
Experiments: A folder for experiments.
Models: A folder containing trained models or model predictions.
Reports: Generated HTML, PDF etc. of your report.
src: Project source code.
README: This file!
.gitignore: Files to exclude from this folder (e.g., large data files).

# Project Guiding Questions

Who is the intended audience for your project?

--Techinical and non-technical management personal of banking sector.

What is the question you will answer with your analysis?

--The aim of the project is to create a model that can predict if a Customer will sign up for a Term load based on their input dataset collection of demographic data, financial data and previous campaign details. This project also aims to export the probability of the same.

What are the key variables and attributes in your dataset?

--Demographic key variables are age, occupation, marital status and education level. Other attribute is customer default data, mortgage data, loan data, last contacted information, the outcome of last campaign.

Do you need to clean your data, and if so what is the best strategy?

--Initial scan of data seems good and we can start with data as is for now.

How can you explore the relationships between different variables?

--Using the model we can see the relationships between demographic and other attributes which can be highly weighted like default data, mortgage data and loan data.

What types of patterns or trends are in your data?

--There coulbe simple patterns like default customers will not take the term loan with being short on savings. There is high probability of pattern for customer contacted in last months who didnt sign up for term deposit will not sign up again if contact in short time like 1 month.

Are there any specific libraries or frameworks that are well-suited to your project requirements?

--we will use data set, pandas, numpy to extract all data. we will use sklearn.model_selection for test and train data to compite and train the model. Additional Details will be added here.

# Who are Stakeholders? and why do they care?

Primary Stakeholders (Direct Users)
--Marketing Team — decides whom to contact
--Campaign Managers — plan outreach strategy
--Sales Agents / Call Center Teams — execute whom to calls

Secondary Stakeholders (Decision Makers)
--Business Executives — care about ROI and revenue
--Finance Team — monitors campaign costs
--Data Science Team — maintains model

Tertiary Stakeholders (Indirect Impact)
--Customers — receive more relevant offers
--Compliance/Risk Teams — ensure ethical targeting

# How to use data set to define stakeholder relevent information

--The dataset is well-structured, complete, and suitable for machine learning modeling. Although it contains no explicit missing values, certain categorical placeholders represent implicit missing data and must be handled carefully. Additionally, variables such as call duration introduce data leakage and should be excluded. Despite limitations such as class imbalance and lack of detailed behavioral financial information, the dataset provides sufficient demographic, economic, and campaign features to successfully build a predictive classification model for term deposit subscription.

# Risks and Uncertainities of data set

Although suitable for modeling, the dataset has some constraints:

### Limited Behavioral Data

The dataset does not include:

- transaction history
- spending patterns
- credit score
- digital activity
  These features could significantly improve prediction quality.

---

### No Time-Series Structure

## The dataset does not capture chronological behavioral trends. Therefore, models cannot learn temporal patterns such as customer behavior changes over time.

### Single Source Dataset

The data comes from one financial institution. Models trained on it may not generalize well to:

- other banks
- different regions
- different economic conditions

---

### No Profit or Revenue Information

The dataset predicts subscription likelihood but does not include financial value indicators such as:

- customer lifetime value
- revenue per customer
- product profitability
  This limits business optimization analysis.

# Methods and Technologies we will use?

# Machine Learning Guiding Details

What are the specific objectives and success criteria for your machine learning model?

How can you select the most relevant features for training?

Are there any missing values or outliers that need to be addressed through preprocessing?

Which machine learning algorithms are suitable for the problem domain?

What techniques are available to validate and tune the hyperparameters?

How should the data be split into training, validation, and test sets?

Are there any ethical implications or biases associated with the machine learning model?

How can you document the machine learning pipeline and model architecture for future reference?
