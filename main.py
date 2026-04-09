import os
import psycopg2
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Esto permite que tu HTML (frontend) hable con el servidor (backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables que configuraremos en el panel de Render
HF_TOKEN = os.getenv("HF_TOKEN")
DB_URL = os.getenv("DB_URL")
MODEL_ID = "rcapia/mistral-7b-lora-contratos"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

class ChatRequest(BaseModel):
    pregunta: str

def get_db_context():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        # Consulta a tu tabla en Neon
        cur.execute("SELECT moneda, SUM(monto) FROM comercia.documentos_cc GROUP BY moneda;")
        filas = cur.fetchall()
        cur.close()
        conn.close()
        return "\n".join([f"- {f[0]}: {f[1]:,.2f}" for f in filas])
    except Exception as e:
        return f"Error de DB: {str(e)}"

@app.post("/consultar")
async def consultar(req: ChatRequest):
    contexto = get_db_context()
    
    # El mismo formato de prompt que usaste en Colab
    prompt = f"<s>[INST] Contexto de Neon:\n{contexto}\n\nPregunta: {req.pregunta} [/INST]"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Llamada a Hugging Face
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code != 200:
        return {"respuesta": "El modelo se está cargando en Hugging Face, por favor reintenta en 30 segundos..."}
        
    res_json = response.json()
    texto_completo = res_json[0]['generated_text']
    respuesta_limpia = texto_completo.split("[/INST]")[-1].strip()
    
    return {"respuesta": respuesta_limpia}
