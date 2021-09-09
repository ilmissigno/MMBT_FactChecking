# MultiModal BiTransformers (MMBT) For Fact-Checking

### Model Training

train.py provides the common training pipeline for all datasets. 
- **task**: fakenewsdet
- **model**: mmbt

The following paths need to be set to start training.

- **configurazione.conf**: Configurazione per il training da li settare i parametri
- **USARE IL DATASET NELLA CARTELLA DATASET SCARICABILE DAL SEGUENTE LINK** : https://drive.google.com/drive/folders/1YDRxCTH_xEMojw1BaEWxDcpLwnkp31f4?usp=sharing

Example command:

```
python main.py -c ./configurazione.conf
```  

### File da consultare

- **mmbt/train.py**: Main file per il training, li avviene il processo di training.
- **mmbt/losses/CrossSimilarity.py**: Implementazione della loss del blocco di CrossSimilarity
- **mmbt/metrics/RankMetrics.py**: Implementazione delle metriche di Ranking per il modello
- **mmbt/data** : Files per il loading del dataset
- **mmbt/models/mmbt.py** : Modello MMBT
