from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Leer token
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("No se encontró el token de Hugging Face en el archivo .env")

print("Token cargado correctamente (no se muestra por seguridad)")

