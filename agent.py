import json
import os, getpass
import datetime
import requests
from typing import List, Optional, TypedDict, Dict, Annotated, Literal
from pydantic import BaseModel, field_validator, ValidationError
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

# =============================================================================
# DEFINIZIONE DELLO STATO CON VERSIONING E MEMORIA PERSISTENTE
# =============================================================================
class VersionedPost(TypedDict):
    version: int              # Versione del post
    timestamp: str            # Data/ora della versione salvata
    content: str              # Testo del post (bozza finale)

class PostRecord(TypedDict):
    topic: str                # Argomento trattato
    category: str             # Categoria (Review, How-to, Evento, ecc.)
    versions: List[VersionedPost]  # Lista delle versioni (per versioning)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # Messaggi di input/output
    prompt: str                   # Messaggio di input dell'utente
    topic: List[str]              # Tema del post attuale
    category: str                 # Categoria: "Review", "How-to", "Evento", ecc.
    query: str                    # Query per la ricerca web
    retrieved_docs: List[str]     # Documenti ottenuti dalla ricerca
    verified_docs: List[str]      # Documenti selezionati dopo verifica
    draft_post: Optional[str]     # Testo della bozza prodotta
    human_feedback: Optional[str] # Feedback dell'utente (human-in-the-loop)
    planning_notes: Optional[str] # Note e report pianificati del processo
    seo_analysis: Optional[str]   # Risultato dell'analisi SEO
    generated_titles: Optional[List[str]]  # Lista di titoli generati
    media_resources: Optional[List[str]]      # Link a risorse multimediali
    previous_posts: List[PostRecord]          # Memoria persistente dei post

MEMORY_FILE = "./blog_memory.json"

def load_memory() -> List[PostRecord]:
    """Carica il file JSON della memoria persistente dei post."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(posts: List[PostRecord]) -> None:
    """Salva la lista dei post (con versioning) su file JSON."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=4, ensure_ascii=False)

llm = ChatOpenAI(model="gpt-4o", temperature=0.8, max_tokens=1500)

# =============================================================================
# AGENTI DEL WORKFLOW
# =============================================================================

# 1. Agente di Web Search che cerca idee per il topic
def tavily_search_ideas_agent(state: AgentState) -> AgentState:
    # Inizializza la ricerca Tavily
    prompt = ( f"Cerca idee per un post nell'ambito gaming di tipo {state['category']}"
              f" e restituisci i risultati in un formato leggibile.\n")
    tavily = TavilySearchResults(max_results=3)
    tavily_results = tavily.invoke(state["query"])
    if tavily_results:
        state["topic"] = [result["title"] for result in tavily_results]
    else:
        state["topic"] = ["Nessun risultato trovato."]
    print("Risultati della ricerca Tavily:")
    for result in state["topic"]:
        print(result)
    state["planning_notes"] += "Risultati della ricerca Tavily:\n"
    for result in state["topic"]:
        state["planning_notes"] += f"- {result}\n"
    return state

# 2. Agente di Web Search per la ricerca di fonti
def web_search_agent(state: AgentState) -> AgentState:
    headers = {"Ocp-Apim-Subscription-Key": "YOUR_BING_API_KEY"}  # Sostituire con la chiave reale
    params = {"q": state["query"], "count": 5, "mkt": "it-IT"}
    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        docs = []
        # Estrae alcuni dettagli per ciascun risultato
        for result in data.get("webPages", {}).get("value", []):
            titolo = result.get("name", "")
            estratto = result.get("snippet", "")
            url = result.get("url", "")
            docs.append(f"{titolo}: {estratto} [Link: {url}]")
        state["retrieved_docs"] = docs
    except Exception as e:
        state["retrieved_docs"] = [f"Errore durante la ricerca web: {str(e)}"]
    return state

# 3. Agente di Verifica delle Informazioni
def verification_agent(state: AgentState) -> AgentState:
    prompt = (
        "Verifica l'accuratezza e l'affidabilità delle seguenti fonti e seleziona quelle più rilevanti e ben documentate:\n"
        f"{state['retrieved_docs']}\n"
        "Rispondi con una lista, una fonte per riga."
    )
    response = llm.invoke(prompt)
    sources = [line.strip() for line in response.content.split("\n") if line.strip()]
    state["verified_docs"] = sources
    return state

# 4. Agente di Redazione della Bozza del Post
def draft_post_agent(state: AgentState) -> AgentState:
    prompt = (
        f"Sei un blogger esperto di tecnologia e devi scrivere un post sul tema '{state['topic']}' "
        f"nella categoria '{state['category']}'. Utilizza le seguenti fonti verificate per scrivere "
        "un articolo strutturato con introduzione, sviluppo e conclusione, lungo circa 400 parole:\n"
        f"{state['verified_docs']}"
    )
    response = llm.invoke(prompt)
    state["draft_post"] = response.content
    return state

# 5. Agente Human-in-the-loop per la Revisione
def human_review_agent(state: AgentState) -> AgentState:
    print("------------ Bozza del Post Generata ------------")
    print(state["draft_post"])
    print("--------------------------------------------------")
    feedback = input("Inserisci eventuali modifiche alla bozza oppure premi INVIO per accettare la versione attuale: ")
    if feedback.strip():
        state["human_feedback"] = feedback.strip()
        state["draft_post"] = feedback.strip()
    else:
        state["human_feedback"] = "L'utente non ha apportato modifiche."
    return state

# 6. Agente di Analisi SEO
def seo_analysis_agent(state: AgentState) -> AgentState:
    prompt = (
        f"Analizza il seguente articolo per identificare opportunità di ottimizzazione SEO. "
        "Fornisci una lista dettagliata di keyword rilevanti, suggerimenti per la densità delle keyword, "
        "e raccomandazioni per migliorare il titolo e il meta description.\n"
        f"Articolo:\n{state['draft_post']}"
    )
    response = llm.invoke(prompt)
    state["seo_analysis"] = response.content
    return state

# 7. Agente di Generazione di Titoli
def title_generation_agent(state: AgentState) -> AgentState:
    prompt = (
        f"Genera 3 titoli accattivanti e ottimizzati per SEO per un post sul tema '{state['topic']}' "
        f"nella categoria '{state['category']}'. I titoli devono essere coinvolgenti e adatti per catturare l'attenzione dei lettori.\n"
        "Rispondi con una lista numerata."
    )
    response = llm.invoke(prompt)
    titles = [line.strip() for line in response.content.split("\n") if line.strip()]
    state["generated_titles"] = titles
    return state

# 8. Agente Media Finder per Contenuti Multimediali (es. immagini da Unsplash)
def media_finder_agent(state: AgentState) -> AgentState:
    headers = {"Authorization": "Client-ID YOUR_UNSPLASH_ACCESS_KEY"}  # Sostituire con la chiave reale
    params = {"query": state["topic"], "per_page": 3}
    media_links = []
    try:
        response = requests.get("https://api.unsplash.com/search/photos", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            link = result.get("urls", {}).get("regular")
            if link:
                media_links.append(link)
    except Exception as e:
        media_links.append(f"Errore nel recupero delle risorse multimediali: {str(e)}")
    state["media_resources"] = media_links
    return state

# 9. Agente di Reportistica
def reporting_agent(state: AgentState) -> AgentState:
    prompt = (
        "Fornisci un report dettagliato che riassuma le fasi del processo editoriale eseguito, "
        "includendo l'analisi SEO, le modifiche apportate tramite revisione umana, "
        "le raccomandazioni dei titoli. Il report deve evidenziare i punti di forza e le aree di miglioramento."
    )
    response = llm.invoke(prompt)
    # Salviamo il report nelle planning_notes
    state["planning_notes"] = response.content
    return state

# 10. Agente per l'Aggiornamento della Memoria Persistente con Versioning
def update_memory_agent(state: AgentState) -> AgentState:
    new_version = {
        "version": 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "content": state["draft_post"]
    }
    exists = False
    # Verifica se per lo stesso topic e categoria esiste già un record
    for record in state["previous_posts"]:
        if record["topic"] == state["topic"] and record["category"] == state["category"]:
            # Incrementa il versioning aggiungendo la nuova versione
            record["versions"].append(new_version)
            exists = True
            break
    if not exists:
        new_record: PostRecord = {
            "topic": state["topic"],
            "category": state["category"],
            "versions": [new_version]
        }
        state["previous_posts"].append(new_record)
    save_memory(state["previous_posts"])
    return state

# =============================================================================
# COSTRUZIONE DEL WORKFLOW CON LANGGRAPH
# =============================================================================
graph = StateGraph(AgentState)
graph.add_node("ideas", tavily_search_ideas_agent)
graph.add_node("search", web_search_agent)
graph.add_node("verify", verification_agent)
graph.add_node("draft", draft_post_agent)
graph.add_node("review", human_review_agent)
graph.add_node("seo", seo_analysis_agent)
graph.add_node("title", title_generation_agent)
graph.add_node("media", media_finder_agent)
graph.add_node("report", reporting_agent)
graph.add_node("memory", update_memory_agent)

# Imposta il flusso sequenziale completo
graph.add_edge(START, "ideas")
graph.add_edge("search")
graph.add_edge("search", "verify")
graph.add_edge("verify", "draft")
graph.add_edge("draft", "review")
graph.add_edge("review", "seo")
graph.add_edge("seo", "title")
graph.add_edge("title", "social")
graph.add_edge("social", "sentiment")
graph.add_edge("sentiment", "media")
graph.add_edge("media", "calendar")
graph.add_edge("calendar", "report")
graph.add_edge("report", "crm")
graph.add_edge("crm", "memory")
graph.add_edge("memory", END)

memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))

# =============================================================================
# ESECUZIONE DEL WORKFLOW
# =============================================================================
if __name__ == "__main__":
    # Carica la memoria persistente dei post precedenti
    stored_posts = load_memory()
    initial_state: AgentState = {
        "topic": "gaming",
        "category": None,
        "query": None,
        "retrieved_docs": [],
        "verified_docs": [],
        "draft_post": None,
        "human_feedback": None,
        "planning_notes": "",
        "seo_analysis": None,
        "generated_titles": None,
        "social_snippets": None,
        "sentiment_analysis": None,
        "media_resources": None,
        "calendar_schedule": None,
        "previous_posts": stored_posts
    }
    
    result = workflow.invoke(initial_state)
    
    # Stampa dei risultati finali
    print("\n------------ RISULTATO FINALE ------------")
    print("Bozza definitiva del post:")
    print(result["draft_post"])
    print("\nAnalisi SEO:")
    print(result["seo_analysis"])
    print("\nTitoli Generati:")
    for title in result["generated_titles"]:
        print(title)
    print("\nSnippet per Social Sharing:")
    for platform, snippet in result["social_snippets"].items():
        print(f"{platform}: {snippet}")
    print("\nAnalisi Sentiment:")
    print(result["sentiment_analysis"])
    print("\nRisorse Multimediali:")
    for media in result["media_resources"]:
        print(media)
    print("\nCalendario di pubblicazione:")
    print(result["calendar_schedule"])
    print("\nReport del processo:")
    print(result["planning_notes"])