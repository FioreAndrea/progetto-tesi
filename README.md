# progetto-tesi

# 04/05
Cose realizzate
- Innanzitutto abbiamo deciso di sviluppare il tutto in Python, lavorando con i dataset utilizzati nel paper
- Abbiamo scelto di utilizzare tutte e 4 le funzioni di distanza presenti in letteratura (Euclidea, Manhattan, Hamming, chisquare) per calcolare i valori di similarità ed avere un riscontro completo quando effettueremo la valutazione dei nostri casi di studio (nel paper invece sono presentati i risultati ottenuti con la sola chisquare con i quali effettuare poi un confronto finale)
- Abbiamo creato le matrici con i valori di similarità tra i moduli di coppie di progetti di versioni diverse (es. Ant 1.3 - Ant 1.4, Ant 1.4 - Ant 1.5 ecc..)

Prossimi sviluppi
- A partire dalle matrici realizzate, selezioneremo i moduli più simili tra di loro (applicando il passo di ottimizzazione, cioè minimizzando e calcolando un threshold, selezionando tutte quelle coppie di moduli i cui valori di similarità sono al di sotto di questa soglia)
- Costruiremo delle nuove matrici formate da 0 e 1 (dove 1 indica una coppia di moduli scelta dalla DS3 ottimizzata). Creeremo delle matrici per ogni funzione di distanza e per ciascun coppia di versioni diverse di ogni dataset

# 16/05
- Implementazione dell'algoritmo ADMM con gestione efficiente dei vari parametri di ottimizzazione
- DS3 ottimizzata con matrice degli elementi rappresentativi per ogni coppia di dataset di versioni consecutive
- Implementazione LogisticRegression per confronto dei dati con il paper di riferimento
