# RateMyProfessors - Data Gathering and EDA

## Data Files

Large data files (>100MB) are not included in this repository due to GitHub size limits. These include:

- `Phase 1/professors_75346.csv` (687 MB): Collection of 75,346 professor records
- `Phase 1/professors_with_SA_labelling.csv` (543 MB): Professor data with sentiment analysis labels
- `Phase 1/topic_results.csv` (1040 MB): Results from topic modeling

## How to Generate the Data

1. Run `Phase_0.ipynb` to collect data from the RateMyProfessor API
2. Run the sentiment analysis pipeline using `sa_pipeline.py`
3. Run the topic modeling code in `topic_modelling.py`

## Alternative Data Access

If you need access to the processed data files, please contact me at williamh.otieno@gmail.com.