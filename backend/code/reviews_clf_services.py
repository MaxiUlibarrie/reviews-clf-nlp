from fastapi import FastAPI
import uvicorn
from models import Review, Reviews_clf

app = FastAPI()

review_clf = Reviews_clf()

@app.get('/')
def health_check():
    return {"status":"I'm up :)"}

@app.get('/classify-review')
def classify_review(review: Review):
    review_type = review_clf.classify_review(review.text)
    try: 
        return {"result" : review_type}
    except Exception as e:
        print(e)
