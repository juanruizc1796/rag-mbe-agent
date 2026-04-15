# MBE RAG Agent

Agente conversacional de **Medicina Basada en la Evidencia (MBE)** construido con:

- **LangGraph** — Orquestación del agente como grafo condicional
- **Llama 3.1 8B** vía **Ollama** — LLM local con tool calling nativo
- **BioBERT** (HuggingFace) — Embeddings biomédicos
- **FAISS** — Vector store para retrieval semántico
- **PostgreSQL** — Memoria conversacional + logs técnicos
- **FastAPI** — API REST

---

## Estructura del Proyecto

```
rag-mbe-agent/
│
├── agent/
│   ├── __init__.py
│   ├── tools.py          # @tool RAG definido con LangChain
│   └── helpers.py        # Detección de idioma, prompts, retry, LLM factory
│
├── utils/
│   ├── __init__.py
│   ├── config.py         # Settings centrales via pydantic-settings + .env
│   ├── db.py             # Capa async PostgreSQL (historial + logs)
│   ├── embeddings.py     # BioBERT wrapper compatible con LangChain
│   └── rag.py            # Gestión del índice FAISS + ingestion + retrieval
│
├── tests/
│   ├── __init__.py
│   └── test_agent.py     # Tests unitarios (sin dependencias externas)
│
├── scripts/
│   ├── ingest.py         # CLI para ingestión de PDFs
│   └── init_db.sql       # SQL de inicialización para PostgreSQL
│
├── data/
│   ├── pdfs/             # PDFs académicos
│   └── faiss_index/      # Índice FAISS persistido (auto-generado)
│
├── graph.py              # Grafo LangGraph completo con todos los nodos
├── main.py               # FastAPI app + endpoints
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

# Arquitectura del Agente

```
START
  │
  ▼
detect_language          ← Detecta ES/EN + carga historial de PostgreSQL
  │
  ▼
classify_intent          ← LLM clasifica: MBE o NON_MBE
  │
  ├─[NON_MBE]──► rejection ──────────────────────────────────┐
  │                                                           │
  └─[MBE]──► query_rewriting     ← Reescribe con PICO (EN)   │
                │                                             │
                ├─[use_rag=False]──► memory_response ─────────┤
                │                                             │
                └─[use_rag=True]──► rag_tool                  │
                                       │                      │
                                       ▼                      │
                                  validate_retrieval           │
                                       │                      │
                                       ├─[insuficiente]──► insufficient_retrieval ─┤
                                       │                                            │
                                       └─[suficiente]──► generate_response ─────────┤
                                                                                     │
                                                         logging (PostgreSQL) ◄──────┘
                                                              │
                                                             END
```

### Descripción de Nodos

| Nodo | Responsabilidad |
|---|---|
| `detect_language` | Detecta ES/EN con heurística + carga últimos N mensajes |
| `classify_intent` | Clasifica si la query es MBE o fuera de dominio |
| `rejection` | Responde rechazando educadamente en el idioma del usuario |
| `query_rewriting` | Mejora la query aplicando PICO, traduce a inglés para retrieval |
| `rag_tool` | Ejecuta retrieval en FAISS con embeddings BioBERT |
| `validate_retrieval` | Verifica si max_score ≥ threshold |
| `insufficient_retrieval` | Informa al usuario que reformule su query |
| `generate_response` | Genera respuesta estructurada con el contexto recuperado |
| `memory_response` | Responde solo con historial (sin RAG) |
| `logging` | Persiste mensajes + log técnico en PostgreSQL |

---

## Setup con Docker (Recomendado)

### Prerrequisitos

- Git: Cualquier versión
- Python ≥ 3.11
- Docker ≥ 24
- Docker Compose ≥ 2.20 (Solo instalanado Docker Compose se instala Docker)
- Al menos **16 GB RAM** (BioBERT + Llama 3.1 8B)
- ~10 GB de espacio en disco (modelos)

### Paso 1 — Clonar y configurar

```bash
git clone <repositorio> 
cd rag-mbe-agent

# Crear archivo de entorno
cp .env.example .env

# Editar si necesitas cambiar credenciales u otros valores
nano .env
```

### Paso 2 — Agregar tus PDFs

Coloca los PDFs académicos de MBE adicionales en caso que sea necesario en data/pdfs.


### Paso 3 — Levantar todos los servicios

Debe tener abierto el Docker Desktop
```bash
docker compose up --build
```

Esto levantará:
- `mbe_postgres` — PostgreSQL en puerto 5432
- `mbe_ollama` — Ollama en puerto 11434
- `mbe_ollama_pull` — Descarga automática de `llama3.1:8b` (se ejecuta una vez)
- `mbe_app` — FastAPI en puerto 8000

>  La primera vez puede tardar 10–20 minutos por la descarga del modelo (~4.7 GB).

### Paso 4 — Verificar que todo esté funcionando

```bash
# Health check
curl http://localhost:8000/health

# Respuesta esperada:
# {"status":"ok","model":"llama3.1:8b"}
```

### Paso 5 — Ingestar los PDFs

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_dir": "/app/data/pdfs"}'
```

O usando el script CLI:

```bash
docker exec mbe_app python scripts/ingest.py --pdf-dir /app/data/pdfs
```

### Paso 6 — Hacer una consulta

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuál es la diferencia entre sensibilidad y especificidad en pruebas diagnósticas?",
    "session_id": "sesion-001"
  }'
```

---

## Setup Local (Sin Docker)

### Prerrequisitos

- Python 3.11
- PostgreSQL 15+
- Ollama instalado localmente

### Paso 1 — Instalar Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Descargar el modelo
ollama pull llama3.1:8b

# Iniciar el servidor
ollama serve
```

### Paso 2 — Crear entorno virtual e instalar dependencias

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 3 — Configurar PostgreSQL

```bash
# Crear base de datos y usuario
psql -U postgres <<EOF
CREATE USER mbe_user WITH PASSWORD 'mbe_pass';
CREATE DATABASE mbe_agent OWNER mbe_user;
GRANT ALL PRIVILEGES ON DATABASE mbe_agent TO mbe_user;
\c mbe_agent
CREATE EXTENSION IF NOT EXISTS pgcrypto;
EOF
```

### Paso 4 — Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y cambiar POSTGRES_HOST=localhost, OLLAMA_BASE_URL=http://localhost:11434
nano .env
```

### Paso 5 — Ingestar PDFs

```bash
mkdir -p data/pdfs
cp /ruta/a/tus/papers/*.pdf data/pdfs/

python scripts/ingest.py --pdf-dir ./data/pdfs
```

### Paso 6 — Levantar la API

```bash
python main.py
```

La API quedará disponible en `http://localhost:8000`.

---

## API Reference

### `GET /health`

Verificación de estado del servicio.

```json
{"status": "ok", "model": "llama3.1:8b"}
```

---

### `POST /chat`

Enviar una pregunta al agente.

**Request:**
```json
{
  "query": "¿Qué es un meta-análisis?",
  "session_id": "sesion-001"
}
```

**Response:**
```json
{
  "session_id": "sesion-001",
  "response": "Un meta-análisis es...",
  "intent": "MBE",
  "language": "es",
  "rewritten_query": "What is a meta-analysis in evidence-based medicine?",
  "retrieval_sufficient": true,
  "max_score": 0.78,
  "node_path": [
    "detect_language",
    "classify_intent",
    "query_rewriting",
    "rag_tool",
    "validate_retrieval",
    "generate_response",
    "logging"
  ],
  "latency_ms": 4320.5
}
```

---

### `POST /ingest`

Ingestar PDFs y construir el índice FAISS.

**Request:**
```json
{"pdf_dir": "/app/data/pdfs"}
```

**Response:**
```json
{"status": "success", "message": "Ingested PDFs from /app/data/pdfs"}
```

---

### `GET /history/{session_id}`

Obtener historial de conversación.

**Response:**
```json
{
  "session_id": "sesion-001",
  "messages": [
    {"role": "user", "content": "¿Qué es un RCT?", "created_at": "..."},
    {"role": "assistant", "content": "Un RCT es...", "created_at": "..."}
  ]
}
```

---

## Ejemplos de Ejecución

### Consulta MBE en español

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuál es el número necesario a tratar (NNT) para la aspirina en prevención cardiovascular?",
    "session_id": "demo-001"
  }'
```

### Consulta de seguimiento (multi-turn)

```bash
# Segunda pregunta en la misma sesión
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Y cuál es el NNH correspondiente?",
    "session_id": "demo-001"
  }'
```

### Consulta en inglés

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the CONSORT checklist for randomized controlled trials.",
    "session_id": "demo-002"
  }'
```

### Query fuera de dominio (será rechazada)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Quién ganó el Mundial 2022?",
    "session_id": "demo-003"
  }'
```

---

## Ejecutar Tests

```bash
# Sin Docker
python -m pytest tests/ -v

# Con Docker
docker exec mbe_app python -m pytest tests/ -v
```

---

## Variables de Configuración

| Variable | Default | Descripción |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Modelo a usar en Ollama |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | URL del servidor Ollama |
| `EMBEDDING_MODEL` | `dmis-lab/biobert-base-cased-v1.2` | Modelo BioBERT |
| `FAISS_TOP_K` | `5` | Documentos a recuperar |
| `FAISS_SIMILARITY_THRESHOLD` | `0.30` | Umbral mínimo de similitud |
| `SHORT_TERM_HISTORY_N` | `10` | Mensajes en memoria de corto plazo |
| `DEFAULT_LANGUAGE` | `es` | Idioma por defecto |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

---

## Esquema de Base de Datos

### `chat_history`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | UUID | PK auto-generada |
| `session_id` | TEXT | Identificador de sesión |
| `role` | TEXT | `user` / `assistant` / `system` |
| `content` | TEXT | Contenido del mensaje |
| `language` | TEXT | Idioma detectado (`es` / `en`) |
| `created_at` | TIMESTAMPTZ | Timestamp de creación |

### `agent_logs`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | UUID | PK auto-generada |
| `session_id` | TEXT | Identificador de sesión |
| `query_original` | TEXT | Pregunta original del usuario |
| `query_rewritten` | TEXT | Query reescrita para retrieval |
| `retrieved_docs` | JSONB | Documentos recuperados |
| `similarity_scores` | JSONB | Scores de similitud |
| `latency_ms` | FLOAT | Latencia total en ms |
| `tokens_used` | INT | Tokens utilizados |
| `node_path` | JSONB | Ruta de nodos ejecutados |
| `error` | TEXT | Error si ocurrió |
| `created_at` | TIMESTAMPTZ | Timestamp |

---

## Troubleshooting

### Modificaciones y volver a lanzar sistema

docker-compose down
docker-compose up --build
### El modelo no responde / timeout

```bash
# Verificar que Ollama tiene el modelo
docker exec mbe_ollama ollama list

# Re-descargar si es necesario
docker exec mbe_ollama ollama pull llama3.1:8b
```

### Error de conexión a PostgreSQL

```bash
# Verificar que el contenedor está healthy
docker compose ps

# Ver logs de postgres
docker compose logs postgres
```

### El índice FAISS no existe

```bash
# Ejecutar ingestión
curl -X POST http://localhost:8000/ingest \
  -d '{"pdf_dir": "/app/data/pdfs"}' \
  -H "Content-Type: application/json"
```

para verificar dentro del contenedor si existe:

Indice:
```
docker exec mbe_app ls -lh /app/data/faiss_index/
```

Ver PDF's Guardados

```
docker exec mbe_app ls -lh /app/data/pdfs/
```

### Ver logs de la aplicación

```bash
docker compose logs -f app
```

### Exponer a internet el API

brew install cloudflared
cloudflared tunnel --url http://localhost:8000

---

## Extender el Agente

### Agregar una nueva Tool

En `agent/tools.py`:

```python
@tool
def new_tool(query: str) -> dict:
    """Descripción de la tool."""
    # implementación
    return {}

ALL_TOOLS = [rag_tool, new_tool]
```

### Agregar un nuevo Nodo al Grafo

En `graph.py`:

```python
def node_new_step(state: AgentState) -> AgentState:
    # lógica del nodo
    return {**state, "node_path": state["node_path"] + ["new_step"]}

graph.add_node("new_step", node_new_step)
graph.add_edge("existing_node", "new_step")
```

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Orquestación | LangGraph 0.2.x |
| LLM | Llama 3.1 8B / Ollama |
| Embeddings | BioBERT (dmis-lab) |
| Vector Store | FAISS CPU |
| Base de datos | PostgreSQL 16 |
| API | FastAPI + Uvicorn |
| Contenedores | Docker + Compose |
| Lenguaje | Python 3.11 |
