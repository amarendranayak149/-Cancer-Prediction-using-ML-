# Cancer Prediction using Machine Learning

## Overview
This project aims to predict the likelihood of cancer using machine learning algorithms. It utilizes a dataset containing various features related to cancer diagnosis and applies classification techniques to determine if a tumor is malignant or benign.

## Features
- Data preprocessing and exploratory data analysis (EDA)
- Machine learning model training and evaluation
- Web application interface using Streamlit
- Interactive visualizations for data insights

## Technologies Used
- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualization
- Scikit-learn for machine learning
- Streamlit for web application
- pyttsx3 for text-to-speech conversion (if applicable)

## Installation
### Prerequisites
Ensure you have Python installed (preferably version 3.7 or later).

### Install Dependencies
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Application
To launch the Streamlit web application, execute:
```bash
streamlit run app.py
```

### Exploring the Data
An exploratory data analysis (EDA) report is available in `eda.html`, which provides insights into the dataset and feature distributions.

### Model Training & Prediction
The machine learning model is trained using scikit-learn and stored in `cancer_pl.pkl`. Predictions are generated using the trained model when new data is input into the Streamlit interface.

## Dataset
The dataset used for training is available as `cancer_prediction_data (2).csv`. It contains various diagnostic features used to classify tumors.

## File Structure
```
.
├── app.py                  # Streamlit web app
├── Cancer_Prediction_using_ML.ipynb  # Jupyter notebook for model training & analysis
├── cancer_prediction_data.csv  # Dataset
├── cancer_pl.pkl           # Trained model
├── eda.html                # EDA report
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── images/
    ├── innomatics_logo.png
    ├── innomatics-footer-logo.png
```

## Contributing
If you'd like to contribute, please fork the repository and submit a pull request with your improvements.

## License
This project is for educational purposes and does not have a specific license.

## Author
[Amarendra Nayak](https://github.com/your-github-profile)

## Contact
- **Phone:** +91-7008631814
- **Email:** toamarendranayak@gmail.com
