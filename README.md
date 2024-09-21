# Boxing Data Analysis and Match Prediction

This project uses machine learning techniques to predict boxing match outcomes based on historical data. The app is built using **Streamlit** and includes sections for data exploration and predictive modeling.

## Features

- **Data Exploration**: Visualizes boxers' statistics like wins, losses, knockout rates, and more.
- **Predictive Modeling**: Predicts match outcomes using **XGBoost** and **Random Forest** models.

## Technologies Used

- **Python**: Data manipulation and machine learning
- **Streamlit**: Web application framework
- **Matplotlib & Seaborn**: For data visualization
- **XGBoost & Random Forest**: Machine learning models
- **Docker**: For containerization

## How to Run Locally

### Prerequisites

- Python 3.x
- Pip (Python package manager)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Safwane-B/boxing-match-predictor.git
   cd boxing-data-analysis


2. Install dependencies

   ```bash
   pip install -r requirements.txt


3. Run the streamlit app
   
   ```bash
   streamlit run your_app.py


5. Access the app at http://localhost:8501.



## Running with Docker

1. Clone the repository:
   
   ```bash
   git clone https://github.com/Safwane-B/boxing-match-predictor.git
   cd boxing-data-analysis


3. Build the Docker image:

   ```bash
   docker build -t boxing-data-app .




4. Run the Docker container:

   ```bash
   docker run -p 8501:8501 boxing-data-app


5. Access the app at http://localhost:8501.





