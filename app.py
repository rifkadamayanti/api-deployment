import os
import uuid
from datetime import datetime
from typing import Optional
import google.generativeai as genai

import cv2
import numpy as np
import tensorflow as tf

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine
)

from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
    Session
)

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "eye_model_22mei.keras"

IMG_SIZE = 96

DB_URL = "sqlite:///./drowsiness.db"

ALLOWED_TYPES = [
    "image/jpeg",
    "image/png"
]

# =====================================================
# LOAD CASCADE
# =====================================================

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# =====================================================
# DATABASE
# =====================================================

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):

    __tablename__ = "predictions"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    filename = Column(String)

    prediction = Column(String)

    confidence = Column(Float)

    open_probability = Column(Float)

    closed_probability = Column(Float)

    eye_detected = Column(Integer)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


Base.metadata.create_all(bind=engine)

# =====================================================
# LOAD MODEL
# =====================================================

if not os.path.exists(MODEL_PATH):

    raise RuntimeError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded")

# =====================================================
# GENERATIVE AI
# =====================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

    client = genai.GenerativeModel(
        "gemini-flash-latest"
    )
else:
    client = None

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(

    title="Drowsiness Detection API",

    version="2.0",

    description="""
    Drowsiness Detection
    CNN + TensorFlow + Gemini AI
    """
)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"]
)

# =====================================================
# SCHEMA
# =====================================================

class PredictionResponse(BaseModel):

    id:str
    filename:str
    prediction:str
    confidence:float
    open_probability:float
    closed_probability:float
    eye_detected:bool
    created_at:datetime

    class Config:
        from_attributes=True


class PredictionWithAdvice(
    PredictionResponse
):

    ai_advice:str


class StatsResponse(BaseModel):

    total_predictions:int

    total_open:int

    total_closed:int

    total_no_eye:int

    avg_confidence:float


# =====================================================
# HELPERS
# =====================================================

def decode_image(
    file_bytes: bytes
):

    nparr=np.frombuffer(
        file_bytes,
        np.uint8
    )

    img=cv2.imdecode(
        nparr,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise ValueError(
            "Image unreadable"
        )

    return img


def detect_eye(
    gray_img
)->Optional[np.ndarray]:

    eyes=EYE_CASCADE.detectMultiScale(

        gray_img,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(20,20)
    )

    if len(eyes)==0:

        return None

    x,y,w,h=eyes[0]

    return gray_img[
        y:y+h,
        x:x+w
    ]


def run_prediction(
    eye_img
):

    img=cv2.resize(
        eye_img,
        (IMG_SIZE,IMG_SIZE)
    )

    img=img/255.0

    img=np.expand_dims(
        img,
        axis=(0,-1)
    )

    pred=float(

        model.predict(
            img,
            verbose=0
        )[0][0]
    )

    open_prob=pred

    closed_prob=1-pred

    prediction=(
        "Open"
        if open_prob>closed_prob
        else "Closed"
    )

    confidence=max(
        open_prob,
        closed_prob
    )

    return{

        "prediction":prediction,

        "confidence":round(
            confidence,
            4
        ),

        "open_probability":round(
            open_prob,
            4
        ),

        "closed_probability":round(
            closed_prob,
            4
        )
    }


def generate_advice(prediction, confidence):

    # cek client kebentuk atau engga
    print("CLIENT:", client)

    if client is None:

        print("USING FALLBACK: client None")

        if prediction == "Closed":
            return (
                "Mata terdeteksi tertutup. "
                "Disarankan beristirahat."
            )

        return (
            "Mata terdeteksi terbuka. "
            "Tetap fokus dan jaga kondisi tubuh."
        )

    try:

        print("USING GEMINI")

        prompt = f"""
        Status mata: {prediction}

        Confidence:
        {confidence*100:.2f}%

        Berikan saran singkat maksimal 3 kalimat.

        Jika Closed:
        beri peringatan.

        Jika Open:
        beri motivasi.
        """

        response = client.generate_content(
            prompt
        )

        print("GEMINI RESPONSE:", response.text)

        return response.text

    except Exception as e:

        print("GEMINI ERROR:", str(e))

        return f"Gemini Error: {str(e)}"
        
def save_prediction(
    db:Session,
    filename,
    result,
    eye_detected
):

    rec=PredictionRecord(

        filename=filename,

        prediction=result[
            "prediction"
        ],

        confidence=result[
            "confidence"
        ],

        open_probability=result[
            "open_probability"
        ],

        closed_probability=result[
            "closed_probability"
        ],

        eye_detected=int(
            eye_detected
        )
    )

    db.add(rec)

    db.commit()

    db.refresh(rec)

    return rec


# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def root():

    return{

        "message":
        "Drowsiness API v2",

        "docs":
        "/docs"
    }


@app.get("/health")
def health():

    return{

        "status":"healthy",

        "model_loaded":True
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse
)

async def predict(
    file:UploadFile=File(...)
):

    if file.content_type not in ALLOWED_TYPES:

        raise HTTPException(

            status_code=400,

            detail="Only jpg/png"
        )

    content=await file.read()

    gray=decode_image(
        content
    )

    eye=detect_eye(
        gray
    )

    if eye is None:

        raise HTTPException(

            status_code=400,

            detail=
            "Eye not detected"
        )

    result=run_prediction(
        eye
    )

    db=SessionLocal()

    try:

        record=save_prediction(

            db,

            file.filename,

            result,

            True
        )

        return record

    finally:

        db.close()


@app.post(
    "/api/v1/predict-with-advice",
    response_model=
    PredictionWithAdvice
)

async def predict_with_advice(
    file:UploadFile=File(...)
):

    base=await predict(
        file
    )

    try:

        advice=generate_advice(

            base.prediction,

            base.confidence
        )

    except Exception as e:
        
        advice=f"AI Error: {str(e)}"

    return {
    "id": base.id,
    "filename": base.filename,
    "prediction": base.prediction,
    "confidence": base.confidence,
    "open_probability": base.open_probability,
    "closed_probability": base.closed_probability,
    "eye_detected": base.eye_detected,
    "created_at": base.created_at,
    "ai_advice": advice
}


@app.get(
    "/api/v1/history"
)

def history():

    db=SessionLocal()

    try:

        data=(

            db.query(
                PredictionRecord
            )

            .order_by(
                PredictionRecord.created_at.desc()
            )

            .all()
        )

        return data

    finally:

        db.close()


@app.get(
    "/api/v1/stats",
    response_model=
    StatsResponse
)

def stats():

    db=SessionLocal()

    try:

        records=db.query(
            PredictionRecord
        ).all()

        total=len(
            records
        )

        return{

            "total_predictions":
            total,

            "total_open":
            sum(
                r.prediction=="Open"
                for r in records
            ),

            "total_closed":
            sum(
                r.prediction=="Closed"
                for r in records
            ),

            "total_no_eye":
            sum(
                not r.eye_detected
                for r in records
            ),

            "avg_confidence":

            round(

                np.mean(

                    [
                        r.confidence
                        for r in records
                    ]

                ),

                4

            ) if total else 0
        }

    finally:

        db.close()
