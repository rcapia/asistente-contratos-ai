from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permitir que tu HTML desde GitHub acceda al Space
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model_path = hf_hub_download(repo_id="rcapia/mistral-7b-contratos-GGUF", filename="contratos_Q4_K_M.gguf")
llm = Llama(model_path=model_path, n_ctx=2048)

class Consulta(BaseModel):
    pregunta: str

@app.post("/consultar")
async def consultar(item: Consulta):
    output = llm(f"[INST] {item.pregunta} [/INST]", max_tokens=256, stop=["[/INST]", "</s>"])
    respuesta = output["choices"][0]["text"].strip()
    return {"respuesta": respuesta}
