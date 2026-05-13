from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from prometheus_client import Counter, generate_latest
from starlette.responses import Response

import pandas as pd
import joblib


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# ---------------- PROMETHEUS METRICS ----------------

REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total App Requests"
)

RECOMMEND_COUNT = Counter(
    "recommend_requests_total",
    "Total Recommendation Requests"
)

# ----------------------------------------------------


# ---------------- DATABASE ----------------

DATABASE_URL = "sqlite:///./songs.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Song(Base):

    __tablename__ = "songs"

    id = Column(Integer, primary_key=True, index=True)

    song_name = Column(String)

    artist = Column(String)

    genre = Column(String)


Base.metadata.create_all(bind=engine)

# ----------------------------------------------------


# ---------------- LOAD MODEL FILES ----------------

similarity = joblib.load("model/similarity.pkl")

df = joblib.load("model/songs.pkl")

# ----------------------------------------------------


# ---------------- HOME ----------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    REQUEST_COUNT.inc()

    db = SessionLocal()

    songs = db.query(Song).all()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "songs": songs
        }
    )

# ----------------------------------------------------


# ---------------- ADD SONG PAGE ----------------

@app.get("/add", response_class=HTMLResponse)
def add_song_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="add_song.html",
        context={}
    )

# ----------------------------------------------------


# ---------------- ADD SONG ----------------

@app.post("/add", response_class=HTMLResponse)
def add_song(
    request: Request,
    song_name: str = Form(...),
    artist: str = Form(...),
    genre: str = Form(...)
):

    db = SessionLocal()

    # Check duplicate
    existing_song = db.query(Song).filter(
        Song.song_name == song_name
    ).first()

    if existing_song:

        return HTMLResponse(f"""
            <h2>
                Song already exists!
            </h2>

            <a href="/add">
                Add Another Song
            </a>

            <br><br>

            <a href="/">
                Go Home
            </a>
        """)

    try:

        song = Song(
            song_name=song_name,
            artist=artist,
            genre=genre
        )

        db.add(song)

        db.commit()

        return HTMLResponse(f"""
            <h2>
                Song '{song_name}' added successfully!
            </h2>

            <a href="/add">
                Add Another Song
            </a>

            <br><br>

            <a href="/">
                Go Home
            </a>
        """)

    except Exception as e:

        return HTMLResponse(f"""
            <h2>
                Error adding song
            </h2>

            <p>{str(e)}</p>

            <a href="/add">
                Try Again
            </a>
        """)
# ----------------------------------------------------


# ---------------- DELETE SONG ----------------

@app.get("/delete/{id}")
def delete_song(id: int):

    db = SessionLocal()

    song = db.query(Song).filter(Song.id == id).first()

    if song:

        db.delete(song)

        db.commit()

    return RedirectResponse(
        url="/",
        status_code=303
    )

# ----------------------------------------------------


# ---------------- RECOMMEND ----------------

@app.get("/recommend/{song_name}", response_class=HTMLResponse)
def recommend(song_name: str, request: Request):

    RECOMMEND_COUNT.inc()

    try:

        matches = df[
            df['track_name']
            .str.lower()
            .str.contains(song_name.lower(), na=False)
        ]

        if matches.empty:

            return HTMLResponse(
                "Song not found in ML dataset"
            )

        index = matches.index[0]

        distances = similarity[index]

        song_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        recommendations = []

        for i in song_list:

            recommendations.append(
                df.iloc[i[0]].track_name
            )

        return templates.TemplateResponse(
            request=request,
            name="recommend.html",
            context={
                "song_name": song_name,
                "recommendations": recommendations
            }
        )

    except Exception as e:

        return HTMLResponse(
            f"Error: {str(e)}"
        )

# ----------------------------------------------------


# ---------------- METRICS ----------------

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# ----------------------------------------------------
