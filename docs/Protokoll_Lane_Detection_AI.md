# Protokoll: Lane Detection AI – Einarbeitung

> Persönliches Protokoll zur Einarbeitung in `src/lane_detection_ai` (Smarty Workspace, ROS 2 Jazzy).
> Erstellt am 21.10.2026. Ich ergänze es laufend.

---

## Inhaltsverzeichnis

1. [Überblick: Wo sitzt die Lane Detection?](#1-überblick-wo-sitzt-die-lane-detection)
2. [Kamera-Kalibrierung](#2-kamera-kalibrierung)
3. [RViz2: Visualisierung](#3-rviz2-visualisierung)
4. [Das Modell: UFLDv2 und Backbones](#4-das-modell-ufldv2-und-backbones)
5. [Training: Grundbegriffe](#5-training-grundbegriffe)
6. [Overfitting und L2-Regularisierung](#6-overfitting-und-l2-regularisierung)
7. [Metriken: F1-Score](#7-metriken-f1-score)
8. [Hyperparameter und Optimizer](#8-hyperparameter-und-optimizer)
9. [Hyperparameter-Optimierung mit Optuna](#9-hyperparameter-optimierung-mit-optuna)
10. [PyTorch Lightning](#10-pytorch-lightning)
11. [ML-Dashboard (TensorBoard / „ML-Board“)](#11-ml-dashboard-tensorboard--ml-board)
12. [DGX: Training auf dem GPU-Server](#12-dgx-training-auf-dem-gpu-server)
13. [Kalman-Filter](#13-kalman-filter)
14. [Messungen: Laufzeit pro Bild](#14-messungen-laufzeit-pro-bild)
15. [Offene Fragen ans Team](#15-offene-fragen-ans-team)
16. [Arbeitslog](#16-arbeitslog)

---

## 1. Überblick: Wo sitzt die Lane Detection?

```
Kamera (vimbax) / Simulation
        │  /camera/image/raw
        ▼
camera_preprocessing   → entzerrt + Vogelperspektive (BEV) → /camera/image/bev
        │
        ▼
lane_detection_ai      → Punkte für linke / mittlere / rechte Linie + Trajektorien → /lane_detection/lane
        │
        ▼
pathplanning → controllers → uart (Motor/Lenkung)
```

- **ROS 2**: Framework, in dem viele kleine Programme („Nodes“) über „Topics“ Nachrichten austauschen.
- **Meine Node:** `lane_detection_ai_node.py`. Sie bekommt das BEV-Bild, lässt das neuronale Netz laufen, rechnet die Pixel in Weltkoordinaten um (`bird_to_world`), fittet pro Linie ein Polynom 2. Grades und publiziert das Ergebnis.
- **Aktives Modell:** `models/best_model_bev.pth` mit `config/model_bev_config.py`, Input 128×96, Backbone ResNet-9.
- **Neuestes Modell:** `models/best_model_bev_2026.*`, Input 512×384. **Ist noch nicht eingebunden.**

---

## 2. Kamera-Kalibrierung

### Was ist das?

Eine Kamera bildet die 3D-Welt nicht perfekt ab. Bei der Kalibrierung bestimmt man mathematisch, *wie* genau sie die Welt verzerrt, damit man das herausrechnen kann. Es gibt zwei Teile:

| Teil | Was wird bestimmt | Womit |
|---|---|---|
| **Intrinsisch** | Brennweite, Bildmittelpunkt, **Linsenverzerrung** (gerade Linien erscheinen krumm, „Fischauge“) | Viele Fotos eines **Schachbretts** aus verschiedenen Winkeln (`img/calib/...`) |
| **Extrinsisch** | **Position und Neigung** der Kamera relativ zum Boden bzw. Auto | Ein Schachbrett („Position Board“) liegt an einer **bekannten Stelle** flach auf dem Boden |

### Warum ist das für mich so wichtig?

1. **Vogelperspektive (BEV):** Das BEV-Bild entsteht aus der extrinsischen Kalibrierung. Bei falscher Kalibrierung sind die Linien im BEV schief, verzogen oder nicht parallel.
2. **Weltkoordinaten:** `bird_to_world` rechnet Pixel in Meter um. Ist die Kalibrierung falsch, plant das Pathplanning mit falschen Abständen.
3. **⚠️ Training:** Das Modell wurde auf BEV-Bildern trainiert. Ändert sich die Kalibrierung, etwa weil die Kamera verrutscht ist oder eine neue Kamera montiert wurde, sehen die BEV-Bilder anders aus als die Trainingsdaten. Das Modell wird dann schlechter. Das nennt man **Domain Shift**. Deshalb gilt:
   - **Erst kalibrieren, dann Daten aufnehmen, dann trainieren.**
   - Nach jeder Kamera-Änderung prüfen, ob das BEV noch zu den Trainingsdaten passt.

### Wie mache ich das?

```bash
ros2 launch camera_preprocessing camera_calibration.launch.py debug:=true
```

- Konfiguration in `src/camera_preprocessing/config/config.yaml`: Brennweite, Sensorgröße, Schachbrettgröße (`board_size`, `square_size`)
- Ergebnis landet in `config/calib.bin`
- Mit `chessboard_path:=none` (Standard) wartet die Node auf das Live-Kamerabild, bis sie das Schachbrett findet
- Danach prüfen: `camera_preprocessing` starten und das BEV-Bild anschauen. **Fahrbahnlinien müssen parallel und gerade sein.**

> Hinweis aus der README: In manchen Rosbags heißt das Topic `/camera/image_raw` statt `/camera/image/raw`. Dann umbenennen mit
> `ros2 bag play <bag> --remap /camera/image_raw:=/camera/image/raw`

---

## 3. RViz2: Visualisierung

**RViz2** ist das Standard-Visualisierungstool von ROS 2. Man kann damit live anzeigen:
- **Bilder** (Display-Typ „Image“): z. B. `/camera/image/bev` oder `/lane_detection/debug/image`
- **Marker / Pfade:** Linien, Punkte, Pfeile in 3D, z. B. die geplante Trajektorie
- **TF-Frames:** Wo ist das Auto, wo die Kamera?

```bash
rviz2
```
Dann links unten auf **„Add“** klicken, **„By topic“** wählen und das Topic auswählen.

**Aktueller Stand:** Meine Node publiziert die Linien nur als **Debug-Bild**, nicht als RViz-Marker. Eine mögliche Verbesserung wäre, die Linien zusätzlich als `visualization_msgs/MarkerArray` zu publizieren. Dann sieht man sie in RViz2 in Weltkoordinaten (Meter) statt nur im Bild.

**Für einen schnellen Bildcheck** reicht auch:
```bash
ros2 run rqt_image_view rqt_image_view
```

---

## 4. Das Modell: UFLDv2 und Backbones

### Ultra Fast Lane Detection v2 (UFLDv2)

- Statt jeden Pixel zu klassifizieren (langsam), legt das Netz **feste Zeilen** (*row anchors*) und **feste Spalten** (*col anchors*) über das Bild.
- Pro Anker-Zeile und Linie beantwortet es zwei Fragen:
  - **Existiert** die Linie in dieser Zeile? → `exist_row`
  - **In welcher Zelle** liegt sie? → `loc_row`. Das ist eine Klassifikation über z. B. 50 oder 100 Zellen.
- Ergebnis: pro Linie eine Liste von Punkten. Das geht sehr schnell, weil es nur wenige Ausgaben gibt.
- Paper und Code: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

### Was ist ein Backbone?

Das **Backbone** ist der vordere Teil des Netzes, der aus dem Bild **Merkmale** (Features) extrahiert: Kanten, dann Formen, dann „das ist eine Linie“. Dahinter sitzt der **Head**, der aus den Merkmalen die eigentliche Antwort macht, hier Anker-Positionen.

Bei uns ist das Backbone ein **ResNet** (Residual Network). Die Zahl gibt die Anzahl der Schichten an:

| Backbone | Größe | Geschwindigkeit | Genauigkeit | Kommentar |
|---|---|---|---|---|
| **ResNet-9** (`"9"`) | sehr klein | sehr schnell | ok | **Bei uns aktiv.** Eigene Variante mit Blöcken `[1,1,1,1]`, **ohne vortrainierte Gewichte** |
| ResNet-18 | klein | schnell | gut | Hat ImageNet-Vortraining (`pretrained=True`) |
| ResNet-34 | mittel | mittel | besser | |
| ResNet-50/101 | groß | langsam | am besten | Für das kleine Auto meist zu langsam |

**Trade-off:** größeres Backbone heißt höhere Genauigkeit, aber mehr ms pro Bild. Auf dem Auto zählt Echtzeit, Ziel sind ≥ 30 FPS (≤ 33 ms pro Bild für die ganze Pipeline).

**Pretrained / Transfer Learning:** Ein auf ImageNet (Millionen Fotos) vortrainiertes Backbone „kann schon sehen“. Damit trainiert man schneller und braucht weniger eigene Daten. ResNet-9 hat das nicht. Ein Versuch mit ResNet-18 pretrained wäre spannend.

---

## 5. Training: Grundbegriffe

| Begriff | Bedeutung |
|---|---|
| **Datensatz** | Bilder plus **Labels**, also wo die Linien wirklich sind („Ground Truth“). Bei uns: `dataset/smartrollerz_bev` |
| **Train / Val / Test Split** | Train zum Lernen, Validation zum Vergleichen während des Trainings, Test für die finale ehrliche Bewertung. **Nie mit Val- oder Test-Bildern trainieren!** |
| **Epoch** | Das Modell hat jedes Trainingsbild einmal gesehen. Config: `epoch = 250` |
| **Batch / Batch Size** | So viele Bilder werden gleichzeitig verarbeitet, bevor die Gewichte angepasst werden (`batch_size = 32`). Größer heißt stabiler, braucht aber mehr GPU-Speicher |
| **Loss** | Zahl, die angibt, wie falsch das Modell liegt. Training = Loss minimieren |
| **Gradient** | Richtung, in die man die Gewichte ändern muss, damit der Loss sinkt |
| **Learning Rate (LR)** | Wie groß der Schritt in diese Richtung ist. Zu groß: Training explodiert. Zu klein: Training dauert ewig |
| **Checkpoint** | Gespeicherte Gewichte (`.pth`). „best_model“ = Checkpoint mit der besten Validierungs-Metrik |
| **Augmentation** | Trainingsbilder künstlich verändern (verschieben, drehen, Helligkeit, Kontrast), damit das Modell robuster wird. Siehe `aug_*` in `best_model_bev_2026_config.py` |
| **Inferenz** | Das fertige Modell benutzen (auf dem Auto), ohne Training |

### Modellformate

| Format | Wofür |
|---|---|
| `.pth` | PyTorch-Gewichte, Standard nach dem Training |
| `.onnx` | Austauschformat, läuft mit `onnxruntime`, oft schneller auf der CPU |
| `.hef` | **Hailo** Executable Format, läuft auf einem Hailo-KI-Beschleuniger (NPU). Wird vom aktuellen Code **noch nicht** unterstützt |

---

## 6. Overfitting und L2-Regularisierung

### Overfitting („Auswendiglernen“)

Das Modell lernt die Trainingsbilder **auswendig**, statt das allgemeine Muster „so sieht eine Linie aus“ zu lernen. Auf neuen Bildern ist es dann schlecht.

**So erkennt man es** in den Kurven im Dashboard (siehe Kapitel 11):

```
Loss
 │\
 │ \___  Val-Loss  ← sinkt erst, steigt dann wieder  ⚠️ Overfitting ab hier
 │  \  \__/‾‾‾‾
 │   \____
 │        \_________ Train-Loss ← sinkt immer weiter
 └──────────────────────── Epochen
```

- Train-Metrik wird immer besser, **Val-Metrik stagniert oder wird schlechter.** Das ist Overfitting.
- Beide schlecht ist **Underfitting**: Das Modell ist zu klein oder hat zu kurz trainiert.

**Gegenmittel:**
- Mehr und vielfältigere Daten (andere Strecken, Licht, Reflexionen)
- Augmentation
- **L2-Regularisierung (Weight Decay)**
- Früher aufhören (**Early Stopping**) bzw. den besten Checkpoint nach Val-Metrik nehmen
- Kleineres Modell

### L2-Regularisierung = Weight Decay

- Zum Loss wird eine Strafe für **große Gewichte** addiert: `Loss_gesamt = Loss + λ · Σ w²`
- Das zwingt das Modell, „einfachere“ Lösungen zu finden, die besser verallgemeinern.
- **λ (Lambda)** heißt in unserer Config `weight_decay` (z. B. `5.5e-06`). Zu groß: Underfitting. Zu klein: kein Effekt.

---

## 7. Metriken: F1-Score

### Precision, Recall, F1

Für Lane Detection: Ein vorhergesagter Linienpunkt zählt als **richtig (TP)**, wenn er nah genug (Schwellwert in Pixeln) an einem echten Linienpunkt liegt.

| | Bedeutung |
|---|---|
| **TP** (True Positive) | Linie vorhergesagt, und sie ist wirklich da ✅ |
| **FP** (False Positive) | Linie vorhergesagt, aber da ist keine ❌ (Geisterlinie) |
| **FN** (False Negative) | Linie da, aber nicht erkannt ❌ |

- **Precision** = TP / (TP + FP) → „Wenn ich eine Linie melde, wie oft stimmt sie?“
- **Recall** = TP / (TP + FN) → „Wie viele der echten Linien finde ich?“
- **F1** = 2 · Precision · Recall / (Precision + Recall) → Mittelwert aus beiden (harmonisches Mittel). Zwischen 0 und 1, **höher ist besser.**

**Warum F1 und nicht Accuracy?** Ein Modell, das nie eine Linie meldet, hat Precision „undefiniert“ und Recall 0. F1 bestraft das. Man kann F1 nicht „austricksen“, indem man nur eine der beiden Größen optimiert.

### „Immer vergleichen“

- In der Config steht `best_metric = "val/local_f1"` mit `best_metric_mode = "max"`. Der beste Checkpoint wird also nach dem F1 auf den **Validierungsdaten** gewählt.
- **Regel:** Jedes neue Modell bzw. jede Änderung wird auf **demselben Val/Test-Set** mit F1 verglichen. Nur dann ist der Vergleich fair.
- Zusätzlich messen: Inferenzzeit (ms/Bild), siehe Kapitel 14. Ein Modell mit +1 % F1, aber doppelter Laufzeit ist für das Auto evtl. schlechter.

**Vergleichstabelle (führe ich weiter):**

| Datum | Modell | Backbone | Input | Val-F1 | ms/Bild (Auto) | Bemerkung |
|---|---|---|---|---|---|---|
| | best_model_bev.pth | 9 | 128×96 | ? | ? | aktuell aktiv |
| | best_model_bev.onnx | 9 | 128×96 | ? | ? | |
| | best_model_bev_2026.pth | 9 | 512×384 | ? | ? | Optuna-optimiert |

---

## 8. Hyperparameter und Optimizer

### Parameter vs. Hyperparameter

- **Parameter / Gewichte:** lernt das Modell selbst im Training (Millionen Zahlen in der `.pth`)
- **Hyperparameter:** stelle **ich** vor dem Training ein. Das Modell kann sie nicht lernen.

**Die wichtigsten Hyperparameter in unserer Config:**

| Hyperparameter | Beispielwert | Wirkung |
|---|---|---|
| `learning_rate` | 0.0017 | Schrittgröße, **der wichtigste Wert** |
| `batch_size` | 32 | Bilder pro Schritt |
| `optimizer` | Adam / SGD | Wie die Gewichte aktualisiert werden |
| `weight_decay` | 5.5e-6 | L2-Regularisierung |
| `momentum` | 0.9 | Nur SGD: „Schwung“ aus vorherigen Schritten |
| `scheduler` + `steps` + `gamma` | multi, [100,150], 0.1 | LR wird in Epoche 100 und 150 mit 0.1 multipliziert (feiner werden am Ende) |
| `warmup` / `warmup_iters` | linear, 100 | LR startet klein und steigt an, damit das Training am Anfang nicht explodiert |
| `backbone` | "9" | Netzgröße |
| `train_width/height` | 512×384 | Eingabegröße: größer = genauer, aber langsamer |
| `num_row` / `num_col` | 55 / 40 | Anzahl Anker-Zeilen/-Spalten |
| `num_cell_row/col` | 100 | Auflösung pro Anker |
| `aug_*` | … | Stärke der Augmentation |

### Optimizer

Der **Optimizer** ist der Algorithmus, der aus dem Gradienten die Gewichtsänderung macht.

| Optimizer | Idee | Typisch |
|---|---|---|
| **SGD** (+ Momentum) | Einfacher Schritt gegen den Gradienten, mit „Schwung“ | Robust, generalisiert oft gut, braucht gut abgestimmte LR |
| **Adam** | Passt die LR **pro Gewicht** automatisch an | Konvergiert schneller, guter Default |
| **AdamW** | Adam mit korrekt entkoppeltem Weight Decay | Moderner Standard, einen Versuch wert |

---

## 9. Hyperparameter-Optimierung mit Optuna

**Problem:** Es gibt zu viele Kombinationen, um alle per Hand zu testen.

**Optuna** probiert automatisch Kombinationen aus und lernt dabei, welche Bereiche gut sind.

Aus `best_model_bev_2026_config.py`:
- `optuna_n_trials = 30`: 30 Trainingsläufe mit verschiedenen Hyperparametern
- `optuna_sampler = "tpe"`: **Tree-structured Parzen Estimator**, wählt die nächsten Versuche „klug“ basierend auf bisherigen Ergebnissen (statt zufällig)
- `optuna_pruner = "median"`: bricht einen Lauf **früh ab**, wenn er schlechter als der Median der anderen ist. Spart viel GPU-Zeit
- `optuna_direction = "maximize"`: maximiert den Val-F1
- `optuna_space`: Suchbereich, z. B. LR zwischen 1e-4 und 2e-2 (logarithmisch), Batch Size 16/24/32, Optimizer SGD/Adam

**Ergebnis der Vorgänger:** Trial 23 war am besten. Adam, LR ≈ 0.0017, Batch 32, Weight Decay ≈ 5.5e-6.

> Achtung: In der Config stehen **zwei** „Optuna sampled“-Blöcke untereinander (Trial 15 und 23). Python nimmt den **letzten**, also Trial 23.

---

## 10. PyTorch Lightning

- **PyTorch:** die Deep-Learning-Bibliothek (Tensoren, Netze, Gradienten)
- **PyTorch Lightning:** ein Aufsatz auf PyTorch, der den **Boilerplate-Code** übernimmt:
  - Trainings-Loop, Validierungs-Loop
  - Multi-GPU-Training (wichtig auf der DGX!)
  - Checkpoints speichern, z. B. „speichere den besten nach `val/local_f1`“
  - Logging nach TensorBoard
  - Mixed Precision (schneller durch 16-Bit-Rechnung)

**Struktur:**
```python
class LaneModel(pl.LightningModule):
    def training_step(self, batch, idx):    # 1 Schritt Training → gibt Loss zurück
    def validation_step(self, batch, idx):  # Metriken auf Val-Daten
    def configure_optimizers(self):         # Optimizer + Scheduler

trainer = pl.Trainer(max_epochs=250, accelerator="gpu", devices=4)
trainer.fit(model, train_loader, val_loader)
```

> ⚠️ **Der Lightning-Trainingscode ist NICHT in diesem Repo.** Hier liegen nur Modell, Loss und Metriken. Ich muss das Team nach dem Trainings-Repo fragen.

---

## 11. ML-Dashboard (TensorBoard / „ML-Board“)

Gemeint ist sehr wahrscheinlich **TensorBoard** (oder alternativ **MLflow** bzw. **Weights & Biases**). Bitte beim Team nachfragen, welches Tool genutzt wird.

**Wofür:** Live-Graphen während des Trainings:
- Train-Loss und Val-Loss über die Epochen → **Overfitting erkennen**
- Val-F1 über die Epochen
- Learning Rate (sieht man den Scheduler?)
- Beispielbilder mit eingezeichneten Vorhersagen
- Optuna-Trials vergleichen

```bash
tensorboard --logdir logs/   # dann im Browser: http://localhost:6006
```
Wenn das Training auf der DGX läuft: Port per SSH weiterleiten:
```bash
ssh -L 6006:localhost:6006 <user>@<dgx-adresse>
```

---

## 12. DGX: Training auf dem GPU-Server

- **NVIDIA DGX** = Server mit mehreren starken GPUs (z. B. A100/H100). Training, das auf dem Laptop Tage dauert, braucht dort Stunden.
- **Arbeitsweise (typisch, beim Team bestätigen):**
  1. Per **SSH** einloggen: `ssh <user>@<dgx-adresse>`
  2. Code per `git clone` holen, Datensatz hochladen (`rsync` / `scp`)
  3. Umgebung: meist **Docker-Container** oder conda/venv mit PyTorch + CUDA
  4. Training in **`tmux`** oder **`screen`** starten. Dann läuft es weiter, wenn die SSH-Verbindung abbricht
  5. Evtl. Job-Warteschlange (**Slurm**: `sbatch`, `squeue`), falls sich mehrere die DGX teilen
  6. GPU-Auslastung prüfen: `nvidia-smi`
  7. Fertige `.pth` zurückkopieren und nach `models/` legen
- **Rücksicht:** Nicht alle GPUs blockieren, wenn andere auch trainieren müssen. Vorher absprechen.

---

## 13. Kalman-Filter

### Problem

Das Modell bewertet **jedes Bild einzeln**. Dadurch **zittern** die erkannten Linien von Frame zu Frame. Manchmal fehlt eine Linie für ein paar Bilder ganz (Reflexion, Kreuzung). Das Pathplanning bekommt dann eine unruhige oder leere Trajektorie (aktuell `[0,0,0]`).

### Idee des Kalman-Filters

Ein Kalman-Filter kombiniert zwei Informationsquellen, **gewichtet nach ihrer Unsicherheit**:

1. **Vorhersage:** „Die Linie sieht wahrscheinlich ähnlich aus wie im letzten Bild.“ Das ist das Bewegungsmodell.
2. **Messung:** „Das Netz sagt, die Linie liegt hier.“

Pro Bild läuft das in zwei Schritten:
- **Predict:** Zustand aus dem letzten Bild fortschreiben, die Unsicherheit wächst.
- **Update:** Messung einrechnen. Ist die Messung sicher, wird sie stärker gewichtet, sonst die Vorhersage. Die Unsicherheit sinkt.

**Fehlt eine Messung** (Linie nicht erkannt), macht man nur Predict. Die Linie bleibt also für kurze Zeit sinnvoll erhalten, statt auf 0 zu springen.

### Anwendung bei uns

- **Zustand** = die 3 Polynom-Koeffizienten `[a, b, c]` pro Linie bzw. pro Trajektorie
- **Messung** = Ergebnis von `np.polyfit`
- **Rauschparameter:**
  - `Q` (Prozessrauschen: wie schnell darf sich die Linie ändern?)
  - `R` (Messrauschen: wie sehr traue ich dem Netz?)
- Einbauort: in `execute_prediction` nach dem `polyfit`-Block
- Einfachere Vorstufe zum Testen: **exponentielle Glättung** `coeff = α·neu + (1−α)·alt`

---

## 14. Messungen: Laufzeit pro Bild

### Schon eingebaut: Timer

Die Node misst jede Stufe mit `smarty_utils/timing`. Die Ausgabe erscheint im Log:

| Timer-Name | Was |
|---|---|
| `msg_transport` | ROS-Bild in ein OpenCV-Bild umwandeln |
| `image_transform` | Resize, RGB, Normalisierung |
| `inference` | **Das neuronale Netz**, meist der größte Anteil |
| `pred2coords` | Netzausgabe in Pixelkoordinaten umwandeln |
| `transformation` | Pixel in Weltkoordinaten |
| `polyfit` | Polynom-Fit und Trajektorie |
| `publish` | Nachricht senden |
| `debug_image` | Nur im Debug-Modus |
| `total` | Alles zusammen |

`filter_strength=40` bedeutet: Der Wert wird über die letzten 40 Bilder geglättet.

### ROS-Werkzeuge

```bash
ros2 topic hz /lane_detection/lane          # wie viele Ergebnisse pro Sekunde (FPS)
ros2 topic hz /camera/image/bev             # wie schnell kommen Bilder rein?
```
Kommen mehr Bilder rein, als die Node schafft, werden Bilder verworfen (Queue-Größe 1). Das ist gewollt, damit sie immer das neueste Bild verarbeitet.

### Offline-Benchmark (geplant)

Ein Skript, das alle Bilder aus `models/eval_2025-03-06/` (1192 Frames) durch das Modell schickt und Folgendes misst:
- Durchschnitt, Median und **p95** (95 % der Bilder sind schneller als dieser Wert) in ms
- Erkennungsrate pro Linie (links / Mitte / rechts)
- Vergleich `.pth` vs. `.onnx` vs. 2026-Modell
- **Wichtig:** auf dem **Auto-Rechner (NUC)** messen, nicht nur auf dem Laptop. Und die ersten ~10 Bilder ignorieren (Warm-up).

### Zielwerte

| Größe | Ziel |
|---|---|
| `total` pro Bild | < 20–30 ms |
| Ausgabe-Frequenz | ≈ Kamera-FPS |
| Erkennungsrate Mitte + rechts | möglichst ~100 % auf der Geraden |

---

## 15. Offene Fragen ans Team

- [ ] Wo liegt der **Trainingscode** (Lightning + Optuna)?
- [ ] Wo liegt der **Datensatz** `smartrollerz_bev`, wie wird gelabelt (welches Tool)?
- [ ] **DGX-Zugang:** Adresse, Account, Docker oder Slurm?
- [ ] Welches **Dashboard**: TensorBoard, MLflow oder W&B?
- [ ] Gibt es **Rosbags** von der Strecke (NAS)?
- [ ] Ist auf dem Auto ein **Hailo**-Beschleuniger verbaut (für `.hef`)?
- [ ] Wurden die 2026-Trainingsdaten mit der **aktuellen Kalibrierung** erzeugt?
- [ ] Welche **Row-Anchor-Formel** wurde beim 2026-Training benutzt? Der Code hat fest `linspace(100, 1540, num_row) / 1550`.
- [ ] Welche Val-F1-Werte haben die bisherigen Modelle?

---

## 16. Arbeitslog

| Datum | Was gemacht | Ergebnis / Erkenntnis |
|---|---|---|
| 20.10.2026 | Workspace analysiert | Pipeline verstanden, Fehlerliste erstellt |
| 21.10.2026 | Begriffe von den Vorgängern aufgearbeitet | dieses Protokoll |
| | Kamera kalibriert | |
| | Erste Fahrt in der Simulation | |
| | | |
