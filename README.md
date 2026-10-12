📩 AI-Based SMS Spam Detection System

An interactive machine-learning application that classifies SMS messages as Spam or Legitimate (Ham) using Python, Scikit-learn, and Streamlit.

📌 Project Overview

SMS spam messages can contain unwanted advertisements, fraudulent offers, and potentially misleading links. This project uses machine learning and natural language processing techniques to identify spam messages and help users analyse SMS content.

The application provides an interactive dashboard where users can enter a message, view the prediction, explore model evaluation results, and download a CSV report.

🎯 Objectives

- Build a machine-learning model for SMS spam classification.
- Process and analyse a labelled SMS dataset using Pandas.
- Convert text messages into numerical features using CountVectorizer.
- Train a Multinomial Naive Bayes classifier.
- Evaluate model performance using standard classification metrics.
- Create an interactive web dashboard using Streamlit.
- Deploy the application online for easy access.

✨ Features

- SMS Classification: Predicts whether an entered message is spam or legitimate.
- Interactive Dashboard: Simple and user-friendly interface.
- Dataset Analysis: Displays useful information about the dataset.
- Model Evaluation: Presents accuracy, precision, recall, F1-score, and a confusion matrix where implemented.
- CSV Report Download: Allows users to download the supported report.
- Online Deployment: Access the application through a web browser.

🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- GitHub
- Streamlit Community Cloud

🤖 Machine Learning Approach

The application uses a Multinomial Naive Bayes classifier for text classification.

The workflow is:

1. Load the labelled SMS dataset.
2. Prepare and validate the message data.
3. Split the dataset into training and testing sets.
4. Convert SMS text into numerical word-count features using CountVectorizer.
5. Train the classifier using labelled messages.
6. Predict whether new messages are spam or legitimate.
7. Evaluate the model using test data and classification metrics.
8. Display results through the Streamlit dashboard.

📂 Project Structure

sms-spam-detection/
│
├── app.py
├── requirements.txt
├── SMSSpamCollection
└── README.md

⚙️ Installation and Setup

Prerequisites

- Python installed on your computer
- Git (optional, for cloning the repository)
- Visual Studio Code or another Python editor

1. Clone the repository

git clone https://github.com/abhishek-Cee/sms-spam-detection.git

Replace "YOUR-USERNAME" with your GitHub username.

2. Open the project folder

cd sms-spam-detection

3. Create and activate a virtual environment (optional but recommended)

On Windows:

py -m venv .venv
.venv\Scripts\activate

4. Install the dependencies

pip install -r requirements.txt

5. Run the application

streamlit run app.py

The application should open in your browser. If it does not, use the local URL shown in the terminal.

🌐 Live Application

Streamlit Demo: sms-spam-detection-dashboard.streamlit.app

📊 Model Evaluation

The model is evaluated using:

- Accuracy: Overall proportion of correct predictions.
- Precision: How many messages predicted as spam are actually spam.
- Recall: How many actual spam messages are detected.
- F1-score: A balance between precision and recall.
- Confusion Matrix: A summary of correct and incorrect classifications.

Use the current results displayed by the application when reporting model performance.

⚠️ Limitations

- Predictions depend on the quality and coverage of the training dataset.
- Unfamiliar words, slang, and new spam techniques may reduce performance.
- The model can produce false positives and false negatives.
- The application classifies entered text; it does not automatically block SMS messages on a phone.
- Predictions should not be treated as a guarantee that a message is safe or fraudulent.

🚀 Future Improvements

- Compare Naive Bayes with other classification algorithms.
- Explore TF-IDF text features.
- Add batch SMS classification.
- Improve support for different languages.
- Test the model on newer and more diverse datasets.
- Add explanations for individual predictions.

🎓 Project Information

Project Title: AI-Based SMS Spam Detection System
Subject: Python for AI
Project Type: Academic Micro-Project

👨‍💻 Author

Name: Abhishek C
Institution: NSS College of Engineering Palakkad 

📄 Disclaimer

This project is developed for educational purposes. Classification results may be incorrect, so users should independently verify suspicious messages and avoid clicking untrusted links or sharing sensitive information.
