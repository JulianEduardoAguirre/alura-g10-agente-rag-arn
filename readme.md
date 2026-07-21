# Challenge ALURA LATAM - G10 - Tech AI
# ⚛️ Asistente Regulatorio Nuclear — Reactores de Investigación (Argentina)

Este proyecto es una aplicación interactiva desarrollada con **Streamlit** que funciona como un asistente inteligente para la consulta de documentación normativa oficial emitida por la **Autoridad Regulatoria Nuclear (ARN)** de la República Argentina, focalizada en **Reactores de Investigación**.

Ejecutable en [https://alura-g10-agente-rag-arn.streamlit.app/](https://alura-g10-agente-rag-arn.streamlit.app/)
---

## 🎯 Objetivo
Facilitar la búsqueda, interpretación y análisis de normativas técnicas complejas mediante una interfaz conversacional intuitiva, precisa y respaldada estrictamente por documentos oficiales.

---

## 🛠️ Arquitectura y Tecnologías
El sistema combina técnicas modernas de Inteligencia Artificial Generativa y recuperación de información:

* **Arquitectura RAG (Retrieval-Augmented Generation):** El modelo responde a las consultas basándose exclusivamente en el contexto extraído de la base documental cargada, reduciendo la posibilidad de alucinaciones.
* **Base de Datos Vectorial (FAISS):** Búsqueda y recuperación ultrarrápida de fragmentos relevantes mediante *embeddings* vectoriales optimizados con la API de **Cohere**.
* **Gestión de Memoria y Contexto:** Mantenimiento del historial conversacional en la sesión para permitir un diálogo fluido y dar seguimiento a preguntas complejas.
* **Interfaz Personalizada (Streamlit):** Diseño responsivo adaptado visualmente con CSS personalizado para ofrecer una experiencia limpia, accesible e institucional.

---

## 🚀 Instalación y Uso Local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/JulianEduardoAguirre/alura-g10-agente-rag-arn.git](https://github.com/JulianEduardoAguirre/alura-g10-agente-rag-arn.git)
   cd alura-g10-agente-rag-arn

2. **Crear y activar el entorno virtual:**
	python -m venv .venv
	# En Windows:
	.venv\Scripts\Activate.ps1
	# En Mac/Linux:
	source .venv/bin/activate

3. **Instala dependencias:**
  ```bash
	pip install -r requirements.txt

4. **Configurar las variables de entorno:**
	Crea un archivo '.env' en la carpeta raíz del proyecto (ejemplo: .env.example) y copia
	tu clave API de Cohere

4. **Ejecutar la aplicación:**
	```bash
	streamlit run app.py

## 🚀 Ejemplo de interacción con el Agente
- **Usuario:**  ¿Cuáles son los límites de dosis para trabajadores en reactores nucleares en vigencia actualmente?
- **Agente:** Según la Norma AR 10.1.1, los límites de dosis para trabajadores en reactores nucleares de investigación son los siguientes: 
							Dosis efectiva: 20 mSv por año, considerado como el promedio en 5 años consecutivos (100 mSv en 5 años), no pudiendo excederse 50 mSv en ninguno de los años individuales...
							.
							.
							.
							Normas aplicables:

							Norma AR 10.1.1 (Norma Básica de Seguridad Radiológica - Revisión 4)
							Norma AR 4.1.1 (Exposición Ocupacional en Reactores Nucleares de Investigación). 🔍

---

**Creado por Julián Eduardo Aguirre**  
[https://github.com/JulianEduardoAguirre](https://github.com/JulianEduardoAguirre)