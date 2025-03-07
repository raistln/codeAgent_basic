from dotenv import load_dotenv
import os
from agent_basic import MyCodeAgent, LocalModel

def main():
    # Cargar el modelo local
    model = LocalModel()

    # Crear el agente con el modelo local
    agent = MyCodeAgent(model=model)

    print("🤖 Agente activado. Puedes darle instrucciones.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        prompt = input("🗣️ Dime algo: ")
        if prompt.lower() == "salir":
            break
        
        respuesta = agent.run(prompt)
        print(f"🤖 Respuesta: {respuesta}\n")

if __name__ == "__main__":
    main()