# AI Music Recommendation System

A FastAPI-based music recommendation system that suggests similar songs using **TF-IDF vectorization and cosine similarity**.
The project includes CRUD operations, SQLite database integration, Jinja templates, Prometheus monitoring, and Docker-ready deployment structure.

#🚀 Features

* 🎵 Music recommendation using ML (TF-IDF + Cosine Similarity)
* ➕ Add / View / Delete songs (CRUD operations)
* 🗄️ SQLite database integration
* 🌐 FastAPI backend with Jinja templates
* 📊 Prometheus metrics for monitoring
* 🧠 Content-based recommendation engine
* 🐳 Docker support (container-ready)
* 📄 Swagger API documentation (`/docs`)


# 🏗️ Tech Stack

* Python
* FastAPI
* Scikit-learn
* Pandas
* SQLite
* Jinja2
* Prometheus
* Docker



## 📂 Project Structure


music_recommender/
│
├── main.py
├── requirements.txt
├── train_model.py
├── .gitignore
│
├── model/
│   ├── similarity.pkl (NOT uploaded to GitHub)
│   ├── songs.pkl
│
├── templates/
│   ├── index.html
│   ├── add_song.html
│   ├── recommend.html
│
└── songs.db (ignored)



## ⚙️ How It Works

1. Dataset is processed using TF-IDF vectorization
2. Cosine similarity is computed between songs
3. When a song is selected, top similar songs are recommended
4. FastAPI serves the backend + UI templates


## ▶️ Run Locally

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload


Open:

* Home: http://localhost:8000
* Add Song: http://localhost:8000/add
* API Docs: http://localhost:8000/docs
* Metrics: http://localhost:8000/metrics

---

## 🐳 Run with Docker


docker build -t music-recommender .
docker run -p 8000:8000 music-recommender


## 📊 Monitoring

Prometheus tracks:

* Total app requests
* Recommendation requests
  
Endpoint:/metrics


Project Architecture Diagram
                    ┌─────────────────────┐
                    │     User (Browser)  │
                    └─────────┬───────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   FastAPI (main.py)      │
                │  - Routes (/add, /)      │
                │  - Recommendation API    │
                └─────────┬────────────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
│ Jinja Templates │ │ SQLite DB      │ │ ML Model           │
│ (Frontend UI)   │ │ songs.db       │ │ TF-IDF + Cosine    │
└────────────────┘ └────────────────┘ └─────────┬──────────┘
                                                │
                                                ▼
                                   ┌──────────────────────┐
                                   │ similarity.pkl       │
                                   │ (precomputed matrix) │
                                   └──────────────────────┘


## ⚠️ Important Notes

* `similarity.pkl` is excluded due to size (>100MB)
* Run `train_model.py` to regenerate model locally
* Dataset used: Spotify Tracks Dataset (Kaggle)

---

## 📌 Future Improvements

* Add user authentication
* Deploy on cloud (AWS / Render)
* Add collaborative filtering
* Integrate Grafana dashboards

---

## 👨‍💻 Author

Built as a **MLOps + Full Stack ML project** using FastAPI and machine learning recommendation techniques.
