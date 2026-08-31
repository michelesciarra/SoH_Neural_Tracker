# SoH Neural Tracker

Stima e previsione dello **State of Health** (SoH) di batterie agli ioni di litio a partire dai dati
di ciclaggio del NASA Battery Aging Dataset, mediante modelli di deep learning per serie temporali.

Progetto di fine corso per l'insegnamento **SC/0081 Deep Learning**, annualità 2026, Corso di Laurea
in Informatica Applicata e Data Analytics (IADA), Università degli Studi di Cagliari.

Autore: **Michele Sciarra**, matricola 60/79/00289.

---

## Domande di ricerca

**D1 — Confronto tra architetture.** Quale architettura neurale prevede meglio il SoH del ciclo
successivo a partire da una finestra di dieci cicli, e quanto si distingue da una baseline naive che
si limita a ripetere l'ultimo valore osservato?

**D2 — Condizioni operative e velocità di degrado.** Quali condizioni operative sono associate a un
degrado più rapido, e quali di esse rappresentano uno stress operativo anziché una conseguenza del
degrado già avvenuto?

**D3 — Robustezza fuori distribuzione.** Quanto peggiorano i modelli su batterie mai viste in
training, cicliate a temperature ambiente diverse?

---

## Le risposte, in breve

I numeri riportati qui sono quelli prodotti dal notebook 04 e dagli esperimenti supplementari; le
tabelle corrispondenti sono indicate accanto a ciascuna risposta.

**D1 — Nessuna architettura neurale supera la persistenza, e le tre ricorrenti non sono
distinguibili fra loro.** Mediana su dieci ripetizioni, RMSE sul test in distribuzione
(`04_d1_riepilogo_test_id.csv`, `04_d1_confronti_appaiati.csv`):

| | RMSE | Skill score |
|---|---|---|
| Baseline di persistenza | 0,0110 | 0 |
| Baseline a retta sulla finestra | 0,0132 | −0,44 |
| SimpleRNN | 0,0268 | −5,3 |
| GRU | 0,0331 | −8,0 |
| LSTM | 0,0458 | −16,3 |
| MLP | 0,1021 | −85,0 |

Il test dei ranghi con segno separa soltanto l'MLP: fra SimpleRNN, GRU e LSTM le differenze sono
compatibili con la variabilità dell'inizializzazione. Nemmeno la retta sulla finestra batte la
persistenza, cioè a un ciclo di distanza la pendenza recente è rumore amplificato e non
informazione. L'ablazione (`04_d1_ablazione.csv`) mostra che togliere il SoH dagli ingressi
peggiora l'errore del 145%, la durata di scarica del 66%, temperatura e tensione di circa un terzo
ciascuna.

**D2 — Nessuna variabile supera entrambi i test di direzione.** Tutte le grandezze misurate durante
la scarica risultano conseguenze del degrado o informazione già contenuta nello stato corrente
(`04_d2_sintesi.csv`). Sui fattori di protocollo, con il blocco come unità di analisi e sulla
finestra comune di 25 cicli (`04_d2_tasso_degrado_per_gruppo.csv`), il gruppo a 4 °C perde 56,4
punti percentuali di SoH ogni 100 cicli contro i 14,8 del gruppo a 24 °C; i due gruppi intermedi
— 22 °C e 43 °C, a 26,6 e 24,7 — si scambiano di posto passando dalla vita intera alla finestra
comune e non sono quindi distinguibili. La dispersione dentro ciascun gruppo è ampia (il gruppo a
24 °C copre da 3,8 a 32,3), e con dodici blocchi su quattro gruppi nessun confronto è conclusivo.

**D3 — La perdita fuori distribuzione va letta in skill score, non in RMSE.** L'insieme fuori
distribuzione è intrinsecamente più facile — la persistenza vi ottiene 0,0080 contro 0,0110 —
quindi un RMSE più basso non significa maggiore robustezza, e la tabella
`04_d3_confronto_insiemi.csv` segnala esplicitamente i casi in cui le due letture sono discordi. Sul
confronto pulito, cioè le stesse celle prima e dopo il cambio di protocollo
(`04_d3_stesse_celle.csv`), il GRU passa da uno skill score di −8,8 a −3,9, l'LSTM da −31,5 a −8,3,
il SimpleRNN resta positivo in entrambi (0,24 e 0,36) e l'MLP crolla da −44,8 a −132,9.

**Perché nessun modello batte la persistenza a un ciclo.** L'esperimento 06 misura il rapporto fra
il segnale da prevedere e il rumore di misura (`06_orizzonte_segnale_rumore.csv`): vale 0,22 a un
ciclo di distanza, attraversa l'unità a dieci cicli e arriva a 3,36 a cinquanta. Il GRU comincia a
superare la persistenza esattamente dal decimo ciclo di orizzonte
(`06_orizzonte_punto_di_pareggio.csv`). Il risultato di D1 non è quindi un limite delle
architetture, ma del bersaglio a orizzonte breve.

**Stimare il SoH senza misurarlo.** L'esperimento 08 toglie dagli ingressi la capacità e il SoH dei
cicli passati e usa la sola firma della ricarica: otto variabili ottengono uno skill score di 0,84
sul test in distribuzione e 0,87 fuori distribuzione rispetto alla media di addestramento
(`08_carica_stima_riepilogo.csv`). È il riferimento disponibile a chi non misuri nulla, non la
persistenza, e il confronto va letto con questa avvertenza.

---

## Dataset

**NASA Prognostics Center of Excellence — Battery Aging Dataset**, disponibile pubblicamente presso
il repository NASA Prognostics Data. Il notebook `01_preprocessing.ipynb` lo scarica
automaticamente: non è necessario alcun download manuale.

Sono utilizzate **dieci batterie**. Due di esse non sono state ciclate a condizioni costanti: B0042
e B0043 cambiano protocollo dopo il quarantesimo ciclo. L'unità di analisi non è quindi la cella ma
il **blocco di protocollo**, cioè un tratto consecutivo di cicli a temperatura ambiente e corrente
di scarica costanti. Una cella ciclata a condizioni costanti coincide con un solo blocco.

| Blocco | Temperatura | Cicli |
|---|---|---|
| B0005, B0006, B0007 | 24 °C | 168 ciascuna |
| B0018 | 24 °C | 132 |
| B0029, B0030, B0031, B0032 | 43 °C | 40 ciascuna |
| B0042 e B0043, prima parte | 22 °C | 41 ciascuna |
| B0042 e B0043, seconda parte | 4 °C | 71 ciascuna |

**Un limite del dataset, da tenere presente nella lettura dei risultati.** Temperatura e corrente di
scarica cambiano sempre insieme: non esiste un blocco caldo a corrente bassa né uno freddo a
corrente alta. I due fattori sono confusi fra loro, e nessuna analisi condotta su questi dati può
separarne gli effetti. Ciò che D3 misura è l'effetto del cambio di protocollo nel suo complesso.

### Suddivisione dei dati

L'assegnazione avviene **per blocco di protocollo, mai per singola finestra**, così che nessuna
finestra temporale attraversi il confine fra due insiemi e nessuna cella di addestramento compaia in
valutazione.

| Insieme | Blocchi | Protocollo | Finestre |
|---|---|---|---|
| training | B0005, B0006 | 24 °C, 2 A | 316 |
| validation | B0007 | 24 °C, 2 A | 158 |
| test in distribuzione | B0018, prima parte di B0042 e B0043 | 24 e 22 °C, 2 A | 172 |
| test fuori distribuzione | B0029–B0032, seconda parte di B0042 e B0043 | 43 °C, 4 °C | 150 |

Il test fuori distribuzione contiene due condizioni opposte — più calda e più fredda
dell'addestramento — anziché una sola. E poiché B0042 e B0043 compaiono in entrambi i test, il
confronto fra i due avviene in parte sulle stesse celle: la differenza di prestazioni misura allora
il cambio di protocollo e non la variabilità fra un esemplare e un altro.

---

## Struttura delle cartelle

```
SoH_Neural_Tracker/
├── assets/                          proposta approvata e presentazione del progetto
├── data/
│   ├── original/                    archivi grezzi scaricati dal notebook 01, mai modificati
│   └── processed/                   dataset derivati, prodotti dai notebook
│       └── sequences_scarica/       una sottocartella per versione del dataset sequenziale
├── experiments/                     una sottocartella per esecuzione di addestramento
├── notebooks/                       i notebook elencati sotto
├── results/
│   ├── figures/                     grafici esportati
│   ├── tables/                      metriche e tabelle in formato .csv
│   └── predictions/                 predizioni sugli insiemi di valutazione
├── README.md
└── requirements.txt
```

`data/original/` è vuota nella cartella consegnata: viene popolata dal notebook 01 al primo
avvio, che scarica gli archivi dalla fonte NASA.

I dati originali non vengono modificati: ogni trasformazione produce un file nuovo in
`data/processed/`.

### Una nota sulla struttura dei dati

Il notebook 01 estrae anche gli indicatori dalla curva di ricarica e li riferisce all'inizio
vita di ciascun blocco, con la stessa logica applicata ai descrittori di scarica. Sono usati
dall'esperimento 08 e disponibili come configurazione alternativa del notebook 02
(`sequences_carica/`, `sequences_completo/`); l'analisi principale resta sulla configurazione
di scarica, perché su quel compito gli indicatori di ricarica non aggiungono informazione —
lo misura il notebook 08.

### File prodotti in `data/processed/`

| File | Prodotto da | Contenuto |
|---|---|---|
| `battery_cycles.csv` | 01 | un record per ciclo di scarica, con SoH, blocco di protocollo, indicatori di validità e descrittori riferiti all'inizio vita |
| `battery_cycles_extra.csv` | 01 | come sopra, per le celle usate solo nell'esperimento supplementare 05 |
| `sequences_scarica/sequence_dataset_window10.npz` | 02 | finestre temporali dei quattro insiemi, già normalizzate |
| `sequences_scarica/composizione_split.csv` | 02 | composizione degli insiemi corrispondente a quel dataset |

Quando il notebook 02 viene eseguito su una configurazione diversa delle variabili di ingresso, la
versione risultante finisce in una sottocartella propria (`sequences_carica/`,
`sequences_completo/`), così che le versioni non si sovrascrivano e restino distinguibili dal nome.

### Contenuto di `experiments/`

Ogni esecuzione del notebook 03 crea una sottocartella `<data>_<ora>_<modello>` contenente:

| File | Contenuto |
|---|---|
| `config.json` | dataset, variabili usate, iperparametri, numero di ripetizioni, grandezza prevista |
| `model.keras` | il modello con la migliore perdita di validazione fra le ripetizioni |
| `training_log.csv` | perdita e metriche per epoca, per ogni ripetizione |

Le esecuzioni non si sovrascrivono mai: il nome della cartella porta l'orario di lancio. La cartella
consegnata contiene l'esecuzione su cui sono calcolati tutti i risultati riportati sopra.

---

## Ordine di esecuzione

I notebook vanno eseguiti **in ordine numerico**. I primi quattro costituiscono la pipeline
principale; gli ultimi quattro sono esperimenti supplementari che si appoggiano ai file già
prodotti.

**La fase di preprocessing si articola su due notebook:** il 01 estrae e ripulisce i cicli, il 02
costruisce gli insiemi e le sequenze. La suddivisione dei dati, con la motivazione del criterio
adottato, è nella sezione 4 del notebook 02.

### Pipeline principale

| # | Notebook | Cosa fa | Dipende da |
|---|---|---|---|
| 01 | `01_preprocessing.ipynb` | scarica il dataset, estrae i cicli di scarica, definisce il SoH, marca i cicli non rappresentativi, individua i blocchi di protocollo e i tratti contigui, costruisce i descrittori riferiti all'inizio vita | — |
| 02 | `02_sequence_preparation.ipynb` | seleziona le variabili di ingresso, suddivide i dati, normalizza e costruisce le finestre temporali | 01 |
| 03 | `03_training.ipynb` | addestra e confronta MLP, SimpleRNN, LSTM e GRU contro due baseline, dieci ripetizioni per architettura | 02 |
| 04 | `04_evaluation.ipynb` | risponde a D1, D2 e D3 rileggendo gli esperimenti salvati | 01, 02, 03 |

### Esperimenti supplementari, eseguibili in qualunque ordine dopo il 03

| # | Notebook | Cosa indaga | Dipende da |
|---|---|---|---|
| 05 | `05_experiment_data_scaling.ipynb` | effetto della quantità e dell'omogeneità dei dati di addestramento, e applicabilità di un Transformer a questa scala | 01, 03 |
| 06 | `06_experiment_horizon.ipynb` | come cambia il confronto con la baseline allungando l'orizzonte di previsione | 01 |
| 07 | `07_experiment_rul.ipynb` | dimostrazione applicativa: stima del ciclo di superamento della soglia critica di SoH | 01, 03 |
| 08 | `08_experiment_charge.ipynb` | stima del SoH dalla sola curva di ricarica, senza misurare la capacità | 01 |

**Perché l'08 è importante.** Tutta la pipeline principale presuppone che la capacità sia
misurata, il che richiede una scarica completa in condizioni controllate: su una batteria in
servizio non è possibile. Il notebook 08 riformula il compito togliendo dagli ingressi sia
la capacità sia il SoH dei cicli passati, e verifica se la sola curva di ricarica — che in
servizio si osserva sempre — basti a stimare lo stato di salute.

**Perché il 06 è importante.** La risposta a D1 è che nessun modello neurale batte la baseline di
persistenza a un ciclo di distanza. Il notebook 06 verifica l'ipotesi che spiega quel risultato: che
il bersaglio a orizzonte breve sia dominato dal rumore di misura, e che allungando l'orizzonte il
rapporto fra segnale e rumore migliori al punto da rovesciare il confronto. Senza quell'esperimento
la risposta a D1 resterebbe una constatazione senza spiegazione.

**Come leggere il 05.** Le configurazioni eterogenee con cinque e sei blocchi di addestramento
presentano una perdita che non converge: l'RMSE sui dati di **addestramento** resta dell'ordine di
0,86, un valore incompatibile con un apprendimento avvenuto. Sono esecuzioni divergenti, non un
effetto della quantità di dati, e vanno lette come tali: la curva di scaling è interpretabile
soltanto sulle configurazioni che convergono. La divergenza è essa stessa un'informazione sulla
fragilità dell'addestramento a questa scala di dati.

**Come leggere il 07.** Sul test in distribuzione è disponibile un solo blocco che raggiunge la
soglia critica entro i cicli osservati, quindi il notebook produce **una sola stima** e non una
misura di accuratezza. L'errore mediano è di 18 cicli su 15 cicli residui, e la scomposizione
mostra che circa il 90% dipende dal metodo di estrapolazione e non dal modello. È una
dimostrazione di come le previsioni si tradurrebbero in una decisione di manutenzione, non un
risultato quantitativo.

### Tempi indicativi su Google Colab

| Notebook | Tempo |
|---|---|
| 01 | ~10 minuti, dominati dal download del dataset |
| 02 | meno di 1 minuto |
| 03 | ~20 minuti: quattro architetture per dieci ripetizioni ciascuna |
| 04 | ~40 minuti, di cui la maggior parte per il guadagno condizionale di D2 |
| 05 | ~30 minuti |
| 06 | ~15 minuti |
| 07 | meno di 1 minuto |
| 08 | ~15 minuti |

Per una passata rapida durante lo sviluppo si possono ridurre `N_RIPETIZIONI` nel notebook 03 e le
liste `SEMI_ABLAZIONE` e `SEMI_CONDIZIONAMENTO` nel notebook 04. Con meno di sei ripetizioni il test
dei ranghi con segno non raggiunge la significatività statistica, e i notebook lo segnalano.

---

## Scelte metodologiche principali

Le motivazioni per esteso sono nei notebook, accanto al codice che le applica. Qui l'elenco di ciò
che un lettore dovrebbe sapere prima di aprirli.

**Il riferimento di capacità è la mediana dei primi cinque cicli**, non il valore massimo osservato.
Il massimo dipende da cicli futuri e su una cella in servizio non sarebbe calcolabile al momento
della previsione.

**I descrittori entrano riferiti all'inizio vita della cella.** Ciascuno viene diviso per il proprio
valore sui primi cicli dello stesso blocco, come già si fa per la capacità nel definire il SoH; le
grandezze termiche vengono inoltre espresse come riscaldamento rispetto alla temperatura ambiente.
In forma assoluta il loro livello dipende dalla cella e dal protocollo, e su una cella mai vista
finirebbe a decine di deviazioni standard da quanto osservato in addestramento.

**I parametri del protocollo sono esclusi dagli ingressi.** Corrente di scarica e temperatura
ambiente sono scelte dello sperimentatore, costanti entro ogni blocco: in addestramento assumono un
valore solo, quindi il modello non ha modo di apprendere cosa accada quando quel valore cambia.

**Le finestre non attraversano mai un ciclo scartato né un cambio di protocollo.** Unirebbero cicli
distanti nella realtà, e la sequenza descriverebbe un'evoluzione mai avvenuta. La contiguità è
verificata nel punto in cui le sequenze vengono costruite, non a valle.

**Il modello prevede la variazione del SoH fra un ciclo e il successivo**, e la stima finale è
l'ultimo valore osservato più la variazione prevista. È la stessa grandezza riscritta: il SoH vale
fra 0.6 e 1 e cambia di circa due millesimi per ciclo, quindi chiedere alla rete il livello
significa chiederle di riprodurre un numero che è già nei suoi ingressi.

**Il confronto è con due baseline.** La persistenza, che ripete l'ultimo SoH osservato, e la retta
ai minimi quadrati sui dieci valori della finestra, prolungata di un ciclo. La seconda usa tutta la
finestra come i modelli neurali ed è quindi il termine di paragone più equo.

**Ogni architettura è addestrata con dieci ripetizioni** e i risultati sono riportati come mediana e
intervallo interquartile, non come esecuzione singola. Con poche centinaia di finestre di
addestramento la distribuzione fra ripetizioni ha code pesanti, e la mediana è meno sensibile della
media a una singola esecuzione fortunata o fallita. Le differenze fra architetture sono valutate con
il test dei ranghi con segno sulle ripetizioni appaiate.

**Gli indicatori di ricarica sono riferiti all'inizio vita come tutti gli altri.** Il
protocollo di carica impostato è identico su tutte le celle, ma si svolge alla temperatura
ambiente della cella: al freddo la resistenza interna più alta accorcia la fase a corrente
costante e allunga la coda a tensione costante. Il riferimento a inizio vita toglie quello
scostamento e lascia il solo invecchiamento.

**D2 usa strumenti diversi per livelli di analisi diversi.** I fattori di protocollo sono costanti
dentro un blocco e variano solo fra blocchi: lì l'unità di analisi è il blocco e le osservazioni
indipendenti sono dodici, quindi si stima una pendenza per blocco e si confrontano i gruppi. Le
variabili che cambiano dentro un blocco vengono invece analizzate con il modello addestrato, tramite
permutation importance e confronto fra modelli annidati.

---

## Limiti del lavoro

**L'insieme di addestramento è piccolo:** 316 finestre da due soli blocchi di protocollo. È la
causa diretta dell'ampiezza dell'intervallo interquartile fra ripetizioni e della divergenza
osservata nell'esperimento 05.

**Il compito a un passo è dominato dalla persistenza.** Il rapporto segnale-rumore misurato nel
notebook 06 vale 0,22 a un ciclo di distanza: il bersaglio è più piccolo del rumore di misura, e
nessun modello può superare sistematicamente la ripetizione dell'ultimo valore.

**Temperatura e corrente di scarica sono confuse nel disegno del dataset**, quindi D2 e D3 misurano
l'effetto del protocollo nel suo complesso e non quello della sola temperatura.

**Nessuna conclusione causale è possibile.** Le associazioni riportate in D2 sono regolarità
osservate su dodici blocchi; il passaggio da associazione a causa richiederebbe un esperimento
controllato che il dataset non contiene.

---

## Esecuzione su Google Colab

Ogni notebook monta Google Drive e individua da solo la cartella di progetto tramite la funzione
`find_project_root()`, che cerca una cartella contenente sia `notebooks/` sia `data/`. È sufficiente:

1. caricare la cartella `SoH_Neural_Tracker/` sul proprio Google Drive;
2. aprire il notebook desiderato in Colab;
3. eseguire tutte le celle in ordine.

Se la cartella viene rinominata o collocata dentro un'altra cartella, il percorso alternativo va
aggiunto alla lista `candidate_paths` dentro `find_project_root()`.

In esecuzione locale, ad esempio in VS Code, la funzione risale l'albero delle cartelle a partire da
quella corrente e non richiede alcuna configurazione.

---

## Librerie richieste

Le dipendenze sono elencate in `requirements.txt`, con le versioni esatte su cui i notebook sono
stati eseguiti. Su Google Colab sono tutte già presenti e non serve installare nulla. In locale:

```bash
pip install -r requirements.txt
```

Non sono state usate librerie esterne a quelle viste durante il corso, con una sola eccezione
metodologica: la **permutation importance** usata nella sezione D2 del notebook 04 è implementata
direttamente con NumPy, senza dipendenze aggiuntive. La scelta rispetto a Integrated Gradients,
visto nel laboratorio della Settimana 10, è motivata nel notebook stesso: la permutation importance
misura l'effetto di un'intera variabile sull'errore calcolato su tutto l'insieme di valutazione,
mentre Integrated Gradients attribuisce una singola predizione ai suoi ingressi. La domanda D2
richiede la prima.

---

## Materiali esterni

Nessuno. Il dataset viene scaricato automaticamente dal notebook 01 e i modelli addestrati, di
dimensione contenuta, sono inclusi nelle sottocartelle di `experiments/`. Non è necessario alcun
link di download esterno.

---

## Acknowledgements

Questo progetto è stato realizzato da **Michele Sciarra** nell'ambito dell'insegnamento Deep
Learning, annualità 2026, erogato dal Corso di Laurea in Informatica Applicata e Data Analytics
(IADA) dell'Università degli Studi di Cagliari.
