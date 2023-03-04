from fastapi import FastAPI
from models import ReviewPL, Reviews_clf

# Loading Model
review_clf = Reviews_clf()

# Getting the app up
app = FastAPI()

@app.get('/')
def health_check():
    return {"status":"I'm up :)"}

@app.get('/classify-review')
def classify_review(reviewpl: ReviewPL):
    try: 
        review_type = review_clf.classify_review(reviewpl.review)
        return {"result" : review_type}
    except Exception as e:
        print(e)
