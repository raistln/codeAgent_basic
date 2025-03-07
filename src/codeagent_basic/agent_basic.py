from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from smolagents import CodeAgent

class LocalModel:
    """Adaptador para usar un modelo local en lugar de la API de Hugging Face."""
    """Puedo usar un modelo más ligero tipo distilgpt2 o gpt2"""
    
    def __init__(self, modelo_id="google/gemma-2b-it"):
        print(f"Descargando y cargando modelo: {modelo_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_id)
        self.model = AutoModelForCausalLM.from_pretrained(modelo_id)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu"  # Puedes cambiar a "cuda" si tienes GPU
        )

    def complete(self, prompt: str) -> str:
        """Genera texto usando el modelo local."""
        resultado = self.pipeline(
            prompt,
            max_length=300, 
            do_sample=True, 
            temperature=0.7,
            truncation=True  # Añadir esta línea para habilitar el truncamiento explícito
        )
        return resultado[0]["generated_text"]

# Aquí, el agente debe usar el modelo LocalModel
class MyCodeAgent(CodeAgent):
    def __init__(self, tools=None, model=None):
        if tools is None:  # Asegúrate de que `tools` no sea None
            tools = []  # Se le asigna una lista vacía si `tools` es None
        # Inicializamos CodeAgent con el modelo local y las herramientas
        super().__init__(tools=tools, model=model)

    def run(self, prompt: str):
        """Ejecutar la tarea en el agente usando el modelo LocalModel."""
        # Aseguramos que el agente use el método `complete` de `LocalModel`
        return self.model.complete(prompt)
    
    
    
    

# Función para verificar razonamientos y gráficos
"""def check_reasoning_and_plot(final_answer, agent_memory):
    # Lógica de validación del informe final y los gráficos generados
    print(final_answer)
    filepath = "saved_map.png"  # Suponiendo que el gráfico se guarda con este nombre
    assert os.path.exists(filepath), "¡Asegúrate de guardar el gráfico como saved_map.png!"
    
    # Abrimos la imagen generada
    image = Image.open(filepath)
    # Realizamos la validación de la respuesta y el gráfico
    prompt = f"Aquí está el informe final y el gráfico generado: {agent_memory.get_succinct_steps()}. ¿Es correcto?"
    
    # Llamada a un modelo multimodal para verificar la respuesta (puedes adaptarlo según tu modelo)
    print(prompt)
    return True  # Retorna True si todo es correcto"""

"""# Agente Manager
manager_agent = CodeAgent(
    model=modelo_local,  # Usamos el modelo local
    tools=[],
    managed_agents=[],  # Gestiona al agente web
    additional_authorized_imports=["pandas", "numpy", "matplotlib", "geopandas"],  # Importaciones adicionales
    planning_interval=5,  # Intervalo de planificación
    verbosity_level=2,  # Nivel de detalle en los logs
    final_answer_checks=[check_reasoning_and_plot],  # Validación final
    max_steps=15,  # Pasos máximos
    )"""
