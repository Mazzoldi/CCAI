# coding: utf-8
"""
Gaming Blog Assistant - LangGraph Workflow Skeleton
=================================================

Questo file definisce la struttura di un agente LangGraph che automatizza il
workflow illustrato nel diagramma Mermaid. Ogni nodo è implementato come
funzione Python e collegato tramite LangGraph.  Le funzionalità che richiedono
integrazione con servizi esterni (Google Trends, Tavily, verifica fonti,
feedback umano, DB, generazione immagini) sono incapsulate in helper classi o
funzioni "adapter" con interfacce ancora da implementare.

Per eseguire serve:
- langchain                   >= 0.1.0
- langgraph                   >= 0.0.30
- openai                      >= 1.0.0
- tavily-python               (wrapper API Tavily)
- pytrends                    (Google Trends unofficial)
- sqlite3 (standard lib)

Le sezioni TODO indicano dove inserire logica applicativa, chiavi API o
integrazione con il sistema di feedback umano (es. web-app front-end,
notifiche Slack, ecc.).
"""

from __future__ import annotations
import os, getpass
import json
import sqlite3
from pathlib import Path
import argparse
from typing import Any, Dict, List, Union, TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import Image, display
from langgraph.prebuilt import ToolNode
from pytrends.request import TrendReq
from tavily import TavilyClient
from langfuse.callback import CallbackHandler
from langchain.callbacks import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langfuse import Langfuse
from langgraph.checkpoint.sqlite import SqliteSaver

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("LANGFUSE_PUBLIC_KEY")
_set_env("LANGFUSE_SECRET_KEY")
_set_env("LANGFUSE_HOST")

langfuse = Langfuse(
  secret_key=os.environ["LANGFUSE_SECRET_KEY"],
  public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
  host=os.environ["LANGFUSE_HOST"]
)

langfuse_handler = CallbackHandler()
cb_manager = CallbackManager([StreamingStdOutCallbackHandler(), langfuse_handler])

DB_PATH = Path("./articles.sqlite")
conn = sqlite3.connect(DB_PATH)
memory = SqliteSaver(conn)

class Draft(TypedDict):
    title: str
    body: str
    images: List[str]

class State(MessagesState):
    prompt: str
    topic: Optional[str]
    sources: Optional[List[Dict[str, Any]]]
    draft: Optional[Draft]

class GoogleTrendsAdapter:
    def __init__(self, hl: str = "en-US", tz: int = 360):
        self.tr = TrendReq(hl=hl, tz=tz)

    def get_trending(self, country: str = "italy", n: int = 20) -> List[str]:
        df = self.tr.trending_searches(pn=country)
        return df.iloc[:, 0].head(n).tolist()

class TopicClassifier:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def is_gaming(self, topic: str) -> bool:
        prompt = [
            {"role": "system", "content": "Sei un esperto di videogiochi, rispondi solo SÌ o NO."},
            {"role": "user", "content": f"Il termine '{topic}' è un argomento di videogiochi?"}
        ]
        resp = self.llm(prompt)
        return resp.content.strip().upper().startswith("SÌ")

class TavilyAdapter:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = TavilyClient(api_key or os.environ.get("TAVILY_API_KEY"))

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return self.client.search(query, k=k)

class SourceVerifier:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def verify(self, sources: List[Dict[str, Any]]) -> bool:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un fact-checker esperto in videogiochi."),
            ("human", "{sources}\nLe fonti sono affidabili? Rispondi SÌ o NO.")
        ])
        msg = prompt.format_messages(sources=json.dumps(sources, ensure_ascii=False))
        reply = self.llm(msg)
        return "SÌ" in reply.content.upper()

class HumanFeedback:
    @staticmethod
    def request(component: str, payload: Union[str, List[str]]) -> bool:
        print(f"[FEEDBACK] {component}: {payload}\nApprovare? (y/n) → ", end="")
        return input().lower().startswith("y")

class ArticleDB:
    SCHEMA = (
        """CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            body TEXT,
            images TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(self.SCHEMA)
    def save(self, title: str, body: str, images: List[str]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO articles(title, body, images) VALUES (?, ?, ?)",
            (title, body, json.dumps(images))
        )
        self.conn.commit()
        return cur.lastrowid

def get_user_prompt_node(state: State) -> State:
    prompt = input(
        "Inserisci il prompt per l'articolo: "
    )
    state["prompt"] = prompt
    return state

def router_node(state: State) -> str:
    prompt_text = state["prompt"]
    messages = [
        {"role": "system",
         "content": (
             "Sei un agente che decide se un prompt utente contiene già un "
             "topic suggerito. Rispondi con 'search_sources' se include un "
             "topic, altrimenti 'choose_topic'."
         )},
        {"role": "user", "content": f"Prompt: {prompt_text}"}
    ]
    resp = llm(messages)
    decision = resp.content.strip().lower()
    return "search_sources" if decision == "search_sources" else "choose_topic"

def choose_topic_node(state: State) -> State:
    global_tr = trends.get_trending(country="", n=30)
    italy_tr = trends.get_trending(country="italy", n=30)
    combined = list(dict.fromkeys(global_tr + italy_tr))
    for term in combined:
        if classifier.is_gaming(term):
            state["topic"] = term
            print(f"[INFO] Topic selezionato: {term}")
            return state
    raise RuntimeError("Nessun topic gaming individuato automaticamente.")

def select_topic_node(state: State) -> State:
    print("Scegli un argomento tra i seguenti:")
    topics = state["topics"]
    for i, topic in enumerate(topics):
        print(f"{i + 1}. {topic}")
    choice = int(input("Scegli un numero: ")) - 1
    if 0 <= choice < len(topics):
        state["topic"] = topics[choice]
        return state
    else:
        raise ValueError("Scelta non valida.")

def search_sources_node(state: State) -> State:
    topic = state.get("topic") or state["prompt"].split("topic:")[-1].strip()
    state["topic"] = topic
    results = tavily.search(topic, k=15)
    filtered = [r for r in results if r.get("score", 0) >= 0.6]
    state["sources"] = filtered
    return state

def verify_sources_node(state: State) -> Union[str, State]:
    return state if verifier.verify(state["sources"]) else "choose_topic"

def draft_node(state: State) -> State:
    topic = state["topic"]
    sources_text = "\n".join(f"- {s['title']} (→ {s['url']})" for s in state["sources"])
    sys_tmpl = SystemMessagePromptTemplate.from_template(
        "Sei un content writer SEO specializzato in videogiochi."
        " Scrivi un articolo di 800-1000 parole sul tema '{topic}' usando un tono informale ma autorevole."
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        sys_tmpl,
        ("human", "Ecco le fonti:\n{sources}\n\nProcedi con la bozza.")
    ])
    msgs = chat_prompt.format_messages(topic=topic, sources=sources_text)
    body = llm(msgs).content
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "Genera un titolo SEO click-bait max 60 caratteri."),
        ("human", "Topic: {topic}")
    ])
    title = llm(title_prompt.format_messages(topic=topic)).content.strip()
    images = [f"https://source.unsplash.com/1600x900/?{topic.replace(' ', '+')}" for _ in range(3)]
    state['draft'] = {'title': title, 'body': body, 'images': images}
    return state

def feedback_title_node(state: State) -> Union[str, State]:
    return state if feedback.request("titolo", state['draft']['title']) else "draft"

def feedback_article_node(state: State) -> Union[str, State]:
    preview = state['draft']['body'][:500] + '…'
    return state if feedback.request("articolo", preview) else "draft"

def feedback_images_node(state: State) -> Union[str, State]:
    return state if feedback.request("immagini", state['draft']['images']) else "draft"

def save_node(state: State) -> State:
    d = state['draft']
    aid = db.save(d['title'], d['body'], d['images'])
    print(f"Articolo salvato con ID {aid}")
    return state

tools = [choose_topic_node, search_sources_node, verify_sources_node]
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True,
    verbose=True,
    callback_manager=cb_manager
).bind_tools(tools)
trends = GoogleTrendsAdapter(hl="it-IT", tz=120)
classifier = TopicClassifier(llm)
tavily = TavilyAdapter()
verifier = SourceVerifier(llm)
feedback = HumanFeedback()
db = ArticleDB()

builder = StateGraph(State)
builder.add_node("get_user_prompt", get_user_prompt_node)
builder.add_node("router", router_node)
builder.add_node("choose_topic", ToolNode([choose_topic_node]))
builder.add_node("select_topic", select_topic_node)
builder.add_node("search_sources", search_sources_node)
builder.add_node("verify_sources", verify_sources_node)
builder.add_node("draft", draft_node)
builder.add_node("feedback_title", feedback_title_node)
builder.add_node("feedback_article", feedback_article_node)
builder.add_node("feedback_images", feedback_images_node)
builder.add_node("save", save_node)

builder.set_entry_point(START)
builder.add_edge(START, "get_user_prompt")
builder.add_edge("get_user_prompt", "router")
builder.add_conditional_edges("router", router_node)
builder.add_edge("choose_topic", "select_topic")
builder.add_edge("select_topic", "search_sources")
builder.add_edge("search_sources", "verify_sources")
builder.add_edge("verify_sources", "choose_topic", condition=lambda out: out == "choose_topic")
builder.add_edge("verify_sources", "draft", condition=lambda _: True)
builder.add_edge("draft", "feedback_title")
builder.add_edge("feedback_title", "draft", condition=lambda out: out == "draft")
builder.add_edge("feedback_title", "feedback_article")
builder.add_edge("feedback_article", "draft", condition=lambda out: out == "draft")
builder.add_edge("feedback_article", "feedback_images")
builder.add_edge("feedback_images", "draft", condition=lambda out: out == "draft")
builder.add_edge("feedback_images", "save")
builder.add_edge("save", END)

human_feedback_nodes = [
    "select_topic",
    "feedback_title",
    "feedback_article",
    "feedback_images"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gaming Blog Assistant")
    parser.add_argument("prompt", help="You are a helpful assistant for a gaming blog.")
    args = parser.parse_args()
    init_state: State = {"prompt": args.prompt}
    graph = builder.compile(interrupt_before=human_feedback_nodes).with_config(
        llm=llm,
        langfuse=langfuse_handler,
        verbose=True,
        max_iterations=10,
        max_tokens=2000,
        checkpointer=memory,
        callbacks=[langfuse_handler, StreamingStdOutCallbackHandler()],
        timeout=30,
    )
    display(Image(graph.get_graph().draw_mermaid_png()))
    graph.invoke(init_state)