```mermaid
graph
A[START] --> B([Prompt utente])
B --> C([Router])
C -. Scelta .-> D([Google Trends e selezione topic])
C -. Topic già dato .-> F([Ricerca online])
D --> E([Scelta/suggerimento topic])
E -. HTL .-> D
E --> F
F -. Score < threshold .-> D
F --> G([Verifica fonti])
G -. Verifica non superata .-> D
G --> H([Bozza])
H --> I([Titolo clickbait])
H --> K([Articolo])
H --> M([Immagini])
I --> J([Feedback titolo])
J -. HTL .-> I
J --> O([Salvataggio])
K --> L([Feedback articolo])
L -. HTL .-> K
L --> O
M --> N([Feedback immagini])
N -. HTL .-> M
N --> O
O --> P[FINE]
```