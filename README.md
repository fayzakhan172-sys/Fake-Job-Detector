# Fake Job Post Detector

This is a small web app where you paste a job post and it tells you how likely the post is to be fake.

I made it as a learning project. I wanted to go through the whole journey once: clean the data, train a model on text, wrap it in an API, make a simple page for it and then pack it all in Docker.

![Result for a fake looking post](screenshot.png)

![Result for a real looking post](screenshot_real.png)

## Why I picked this

Fake job posts are everywhere, and freshers like us get targeted the most. They usually ask for a registration fee, bank details or Aadhaar. I wanted to see if a model can catch these just from the way the post is written.

## The data

I used the Kaggle dataset [Real or Fake: Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).

- Around 17,880 job posts
- Only about 5% are fake, so the data is quite lopsided
- Some columns are long text (title, company profile, description, requirements) and some are short (employment type, experience, education, and yes/no things like "has company logo")

## What I used

- Python and pandas for the data
- scikit-learn for everything on the ML side: `CountVectorizer`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline` and `LogisticRegression`
- joblib to save the trained model
- FastAPI and Uvicorn for the backend
- Plain HTML, CSS and JavaScript for the page
- Docker to package the app
- Jupyter Notebook in VS Code for all the experiments

## How it works

First I joined the title, company profile, description and requirements into one single text column. Then I split the data into train and test (80/20).

After that, a `ColumnTransformer` treats each kind of column differently:

- the text goes through `CountVectorizer`, which is basically Bag of Words (about 90,978 word features)
- the category columns go through `OneHotEncoder` (28 features)
- the yes/no flags (remote work, company logo, screening questions) are passed as they are (3 features)

All of this sits inside one `Pipeline`, with a `LogisticRegression` at the end (`max_iter=1000` and `class_weight='balanced'` because fake posts are so few). The trained pipeline is saved as `fake_job_post_detector_model.pkl`.

`app.py` loads that file and gives a `/predict` endpoint using FastAPI. `index.html` sends the form data to it and shows the fake probability on the page.

## Results

I tested on 3,576 posts, out of which 181 were fake. These numbers are for the fake class:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.979 | 0.77 | 0.84 | 0.80 |

The accuracy looks great, but that is mostly because 95% of the posts are real anyway. A model that always says "real" would also score high. So I looked at precision, recall and F1 for the fake class instead. Recall of 0.84 means it catches most fake posts. Precision of 0.77 means it sometimes flags a real post by mistake.

## Run it with Docker

```
git clone https://github.com/fayzakhan172-sys/Fake-Job-Detector.git
cd Fake-Job-Detector
docker build -t fake-job-detector .
docker run -p 8000:8000 fake-job-detector
```

Now open http://127.0.0.1:8000 in your browser. On Windows the `0.0.0.0` link does not open, so use `127.0.0.1`.

## Run it without Docker

```
pip install -r requirements.txt
uvicorn app:app --reload
```

## Files in this repo

```
app.py                              FastAPI app (serves the page and /predict)
index.html                          The web page
Fake Job Post Detector.ipynb        Data prep, training and evaluation
fake_job_post_detector_model.pkl    Saved model
requirements.txt                    Python packages
Dockerfile                          Docker setup
screenshot.png                      Result for a fake looking post
screenshot_real.png                 Result for a real looking post
```

## Problems I ran into

- My project folder was inside OneDrive, and pandas broke while installing because some of its files were locked (`Access is denied`). I had to stop the running Python processes and delete the half installed folder by hand.
- For a while the page showed `NaN% risk`. It took me some time to find that I had typed `pobability` instead of `probability` in `app.py`, so the page could not find the value.
- I tried running the `.ipynb` file with the Code Runner extension and got `name 'null' is not defined`. Notebooks have to be run from the notebook's own Run button.
- Some files ended up inside the `venv` folder by mistake, so Docker could not see them until I moved them to the main folder.

## Limitations

- Bag of Words does not understand word order or meaning, it only counts words.
- The dataset is old and mostly English, so newer scam messages or Hinglish posts may slip through.
- The output is only a probability, not proof. A high score just means the post looks like the fake ones the model has seen before.

## What I want to try next

- TF-IDF and word pairs (n-grams), and compare the scores
- Other models like XGBoost or Random Forest against Logistic Regression
- Showing which words pushed a post towards "fake", so the result can be explained
- Training on newer, India specific job posts
- Putting the app online so anyone can try it without installing anything
