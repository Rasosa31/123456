# Stock Price Movement Prediction with Machine Learning

This academic project applies Machine Learning techniques to predict whether a stock's price will go up or down the next day, using historical data and supervised models. The approach is practical, featuring a functional API, Dockerized environment, and a trained model ready to make predictions based on user-provided inputs.

## Project Objective

To develop a binary prediction system (up / down) for stock price movement, based on features extracted from historical data. The model was trained, evaluated, and deployed via an API to facilitate integration.

## Project Structure

### File / Folder	Description

	1. stock_pred_ec_oil.ipynb	Main notebook with data exploration, model training, and evaluation.
	2. app.py	Script exposing the model as a REST API using FastAPI.
	3. test_api.py	Unit tests to validate API functionality.
	4. requirements.txt	List of dependencies required to run the project.
	5. Dockerfile	Configuration for containerizing the application.
	6. DATA/	Folder containing the data used to train the model.
	7. best_model.pkl	Trained and serialized model ready to be loaded by the API.
	8. README.md	Project documentation.

## Technologies Used

    -Python 3.10+
    -Scikit-learn
    -Pandas / NumPy
    -FastAPI
    -Docker
    -Pytest

## Example Usage

Once the API is running, you can send a POST request with the following input features:
  
json

      {
  "Close": 10.0,  
  "Volume": 500000,
  "SMA_100": 12.0,
  "RSI_14": 56.0,
  "WTI_Close": 46.32
       }

The response will be:

json
       {
  "prediction": 1,
  "confidence": 0.6,
  "meaning": "1 = DOWN tomorrow"
        }

## Input Explanation

    1. Close: Closing price of the stock on the last trading day.
    2. Volume: Number of shares traded at the close of the last trading day.
    3. SMA_100: 100-period Simple Moving Average at the close of the last trading day.
    4. RSI_14: Relative Strength Index (14-period) at the close of the last trading day.
    5. WTI_Close: Closing price of WTI crude oil on the last trading day.

## Results

The model achieved an accuracy of 53.1% on the test set, using a classifier [specify: RandomForest, XGBoost, etc.]. It was evaluated using metrics such as accuracy, F1-score, and confusion matrix.

## How to run the prediction model with Docker

The model runs with  Conda and  Docker and the test needs to use the file  test_api.py. Follow the next stpes:

## How to run the prediction model
Follow this steps to clone the repository, built the Docker and get a prediction from the API.

### To clone the repository
```bash
git https://github.com/Rasosa31/midterm_project.git
```

#### cd tu_repositorio

#### Activate the Conda environment (optional if you're only using Docker)

If you want to use the Conda environment locally:
```bash
conda activate nombre_del_entorno
```
Make sure you have the environment set up beforehand. If you don't have it, you can create it with:

```bash
conda create -n nombre_del_entorno python=3.8
```
```bash
conda activate nombre_del_entorno
```
```bash
pip install -r requirements.txt
```

### Built the Docker container
```bash
docker build -t ec-wti-api .
```
This command can take some seconds to complete. The period (.) at the end is mandatory.al final es obligatorio.

### Run the Docker container
```bash
docker run -p 5001:5000 ec-wti-api
```
That start the API en el puerto 5000 inside the container , expose as 5001 in the local machine.

## Test the API from another terminal
Open a second terminal (with the same environment enabled if you're using Conda) and run:
```bash
python test_api.py
```
This will send a request to the API and display a prediction like:
json
{
  "prediction": 0,
  "confidence": 0.72,
  "meaning": "0 = SUBE mañana"
}
## Additional notes

Make sure you have Docker installed: Install Docker
The test_api.py file must be in the repository root or the specified folder.

If port 5001 is in use, you can change it using the `Docker run` command.


## Additional Notes

    NaN values handled via dropna()
    Only 5 inputs required — API computes additional features
    Accuracy: 53.1% (+2.5% improvement with WTI feature)
    Model serialized with joblib (preferred over pickle)

# Credits

This project was developed as part of the Machine Learning Zoomcamp course by DataTalksClub (https://github.com/DataTalksClub/machine-learning-zoomcamp/), for educational and practical exploration in data science and machine learning.