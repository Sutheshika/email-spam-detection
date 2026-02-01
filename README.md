# Email Spam Detection

A machine learning-based web application that classifies emails as spam or legitimate (ham) using a Multinomial Naive Bayes classifier trained on email data.

## Project Overview

This project implements an email spam detection system with two main components:
- **spam.py**: Model training and testing script using machine learning
- **app.py**: Flask web application for making predictions on new emails

## Tech Stack

- **Framework**: Flask (Python web framework)
- **ML Library**: Scikit-learn (machine learning algorithms)
- **NLP**: NLTK (Natural Language Toolkit)
- **Data Processing**: Pandas, Seaborn, Matplotlib
- **Model**: Multinomial Naive Bayes Classifier
- **Frontend**: HTML/CSS (templates)

## Prerequisites

### System Requirements
- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries
- flask
- scikit-learn
- pandas
- numpy
- nltk
- seaborn
- matplotlib

## Environment Setup

### 1. Create a Virtual Environment (Recommended)

**Windows (PowerShell/CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install flask scikit-learn pandas numpy nltk seaborn matplotlib
```

### 3. Download NLTK Data

After installing, you may need to download NLTK corpora. Run the following in Python:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Or uncomment the `nltk.download()` calls in `spam.py` and run it.

## Project Structure

```
email-spam-detection/
├── README.md                 # Project documentation
├── app.py                    # Flask web application
├── spam.py                   # Model training script
├── model.pkl                 # Trained Naive Bayes model (pre-trained)
├── cv-transform.pkl          # Trained CountVectorizer (pre-trained)
├── EmailCollection           # Dataset file (emails with labels)
├── templates/                # HTML templates for web UI
│   ├── home.html            # Main input page
│   └── result.html          # Results page
├── static/                   # Static files (CSS, JS, images)
└── .git/                     # Git repository
```

## How to Use

### Option 1: Run the Web Application

1. **Activate the virtual environment** (if using one):
   ```bash
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

2. **Start the Flask application**:
   ```bash
   python app.py
   ```

3. **Access the application**:
   - Open your web browser and navigate to: `http://127.0.0.1:5000/`
   - Enter an email message in the text field
   - Click the prediction button to classify it as Spam or Ham (legitimate)

4. **View results**:
   - The application will display whether the email is classified as **Spam** or **Ham**

### Option 2: Run the Model Training Script

To retrain the model with the dataset:

```bash
python spam.py
```

This will:
1. Load the email dataset from `EmailCollection`
2. Preprocess and clean the text (tokenization, stemming, stop word removal)
3. Create a Bag of Words model
4. Train a Multinomial Naive Bayes classifier
5. Evaluate accuracy on test data
6. Display predictions for sample emails
7. (Uncomment lines to save the trained model to pickle files)

## Dataset

The project uses the `EmailCollection` file which contains labeled email messages:
- **Format**: Tab-separated values (TSV)
- **Columns**: Label (0=Ham, 1=Spam) | Email message
- **Example**: 
  ```
  0	Go until jurong point, crazy...
  1	WINNER!! You have been selected...
  ```

## Model Details

### Algorithm
- **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' theorem
- **Alpha Parameter**: 0.8 (Laplace smoothing)

### Preprocessing Steps
1. Remove non-alphabetic characters
2. Convert text to lowercase
3. Tokenize into words
4. Remove English stop words
5. Apply Porter Stemming

### Feature Extraction
- **CountVectorizer**: Converts text to numerical features
- **Max Features**: 3500 (limits vocabulary size)

## Performance

The model achieves accuracy on the test dataset (displayed when running spam.py).

## File Descriptions

### app.py
Flask web application that:
- Loads pre-trained model and vectorizer from pickle files
- Provides a web interface for users to input email text
- Processes user input and returns spam/ham prediction
- Serves HTML templates

### spam.py
Training and testing script that:
- Loads and preprocesses email data
- Creates machine learning model
- Trains on labeled data
- Evaluates model performance
- Makes predictions on sample emails
- Optionally saves model to pickle files

### model.pkl
Pre-trained Multinomial Naive Bayes classifier (binary classification)

### cv-transform.pkl
Pre-trained CountVectorizer for text-to-numerical feature conversion

## Common Issues & Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```
Or manually install: `pip install flask scikit-learn pandas nltk seaborn matplotlib`

### Issue: NLTK data files not found
**Solution**: Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Port 5000 already in use
**Solution**: Modify the port in app.py:
```python
app.run(port=5001)  # or any available port
```

### Issue: EmailCollection file not found
**Solution**: Ensure the dataset file is in the project root directory

## Future Improvements

- Add more sophisticated NLP techniques (TF-IDF, Word2Vec)
- Implement additional classifiers (SVM, Random Forest)
- Add email attachment detection
- Implement database for logging predictions
- Add user authentication
- Deploy to cloud platform (Heroku, AWS, Azure)
- Create REST API endpoints
- Add model retraining functionality

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license here]

## Author

[Add author information here]

## Contact

For questions or issues, please contact [add contact information]
