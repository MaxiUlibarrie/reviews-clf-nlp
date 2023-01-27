# Reviews Classifier
This is a project to identify a movie review in spanish as positive or negative.

### Require:
* Docker
* Docker Compose

### CI/CD:
There are two principal processes for this project (both managed using Docker Compose):
* TRAINING
* BACKEND

# TRAINING

## Build Train Image

> `docker-compose build train`

## Run training model

### Parameters:

#### Require:
* **--version-model**: version of the new model to train
* **--epochs**: number of epochs

#### Not Require:
* **--n-classes**: number of classes (default: 2)
* **--max-length-tokens**: Max length of the tokens to be used (default: 300)
* **--batch-size**: batch size for training (default: 16)
* **--random-seed**: random seed for to start training (default: 42)

Example:
> `docker-compose run train --version-model 1 --epochs 5`

### Config file
Also in the configuration file it is posible to change these default values (`common/config.json`)

# BACKEND

## Build Image 

> `docker-compose build backend`

## Getting service up

> `docker-compose up backend`

## Endpoints

All the endpoints are using `localhost:8000`

### /
Health check. It returns:
```
{"status":"I'm up :)"}
```
### /classify-review

Body Example:
```
{ "review": "Esta pelicula es muy buena." }
```

It returns:
```
{ "result" : "POSITIVA" }
```