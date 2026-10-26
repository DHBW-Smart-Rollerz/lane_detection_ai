# Lernpfad: KI trainieren (Lane Detection AI)

> Mein Fahrplan, um zu verstehen, **was** ich trainiere, **wie** ich trainiere und **was** ich ändern soll.
> Begriffe, die hier nur kurz vorkommen, stehen ausführlicher in [Protokoll_Lane_Detection_AI.md](Protokoll_Lane_Detection_AI.md).
> Meine Aufgaben stehen in [Aufgaben_Lane_Detection_AI.md](Aufgaben_Lane_Detection_AI.md).

**Legende:** 🎬 Video · 📄 Text/Paper · 🛠️ selbst ausprobieren · ⚠️ Achtung

---

## Inhaltsverzeichnis

1. [Der Fahrplan in 3 Wochen](#1-der-fahrplan-in-3-wochen)
2. [Das große Bild: Was passiert beim Training?](#2-das-große-bild-was-passiert-beim-training)
3. [Videos und Lesestoff](#3-videos-und-lesestoff)
4. [Was soll ich trainieren? (Die richtige Reihenfolge)](#4-was-soll-ich-trainieren-die-richtige-reihenfolge)
5. [Hyperparameter: Was bedeutet was, und was passiert bei Änderungen?](#5-hyperparameter-was-bedeutet-was-und-was-passiert-bei-änderungen)
6. [Grad-CAM und HiResCAM: Ins Modell hineinschauen](#6-grad-cam-und-hirescam-ins-modell-hineinschauen)
7. [Experimentplan: Meine ersten Trainingsläufe](#7-experimentplan-meine-ersten-trainingsläufe)
8. [Training auf deep-thought (DGX) starten](#8-training-auf-deep-thought-dgx-starten)
9. [Checkliste](#9-checkliste)

---

## 1. Der Fahrplan in 3 Wochen

| Woche | Ziel | Was ich mache |
|---|---|---|
| **1** | Verstehen, wie ein Netz lernt | Videos aus Stufe 1 und 2 (Kapitel 3). Fehlertabelle (Aufgabe 1) ausfüllen. Grad-CAM auf den Eval-Bildern laufen lassen (Kapitel 6). |
| **2** | Verstehen, wie man *gut* trainiert | Videos aus Stufe 3. Karpathys „Recipe“ lesen. Trainingscode im Branch `origin/training` durchgehen. Zugang zur DGX klären. |
| **3** | Erster eigener Trainingslauf | Baseline nachtrainieren (Experiment 0, Kapitel 7). Ergebnis in MLflow anschauen. Eine Sache ändern, vergleichen. |

**Wichtigste Regel:** Ich muss nicht alles verstehen, bevor ich anfange. Videos schauen → etwas ausprobieren → zurück zu den Videos. So bleibt es hängen.

---

## 2. Das große Bild: Was passiert beim Training?

Training ist eine Schleife, die tausende Male läuft:

```
     ┌─────────────────────────────────────────────────────────────┐
     │                                                             │
     ▼                                                             │
1. Batch holen          32 BEV-Bilder + ihre gelabelten Linien     │
     │                                                             │
2. Forward Pass         Netz macht eine Vorhersage                 │
     │                  (wo liegen die 3 Linien?)                  │
     │                                                             │
3. Loss berechnen       Wie falsch war die Vorhersage?             │
     │                  (eine Zahl, je kleiner desto besser)       │
     │                                                             │
4. Backward Pass        Für jedes Gewicht ausrechnen: In welche    │
     │                  Richtung muss es sich ändern? (Gradient)   │
     │                                                             │
5. Optimizer-Schritt    Gewichte ein kleines Stück verschieben     │
     │                  (wie groß = Learning Rate)                 │
     └─────────────────────────────────────────────────────────────┘

Einmal durch alle Trainingsbilder = 1 Epoche. Bei uns: 250 Epochen.
Nach jeder Epoche: Validierung auf Bildern, die das Netz NIE zum Lernen sieht → val/local_f1
```

**Bei uns konkret (UFLDv2):**

```
BEV-Bild (128×96)
   │
   ▼
Backbone (ResNet-9)        layer1 → layer2 → layer3 → layer4
   │                       „Augen“: erkennt Kanten → Striche → Linienstücke
   │                       Jede Stufe halbiert die Auflösung (am Ende nur noch 4×3!)
   ▼
pool (1×1 Conv, 512 → 8 Kanäle)
   │
   ▼
flatten + MLP (2 Linear-Schichten)
   │                       „Gehirn“: entscheidet, in welcher Zelle jede Linie liegt
   ▼
loc_row / loc_col          WO liegt die Linie in jeder Anker-Zeile/-Spalte?
exist_row / exist_col      GIBT es die Linie in dieser Anker-Zeile/-Spalte?
```

⚠️ **Wichtige Beobachtung:** Bei 128×96 Pixeln Eingang bleiben nach `layer4` nur **4×3 Positionen × 8 Kanäle = 96 Zahlen** übrig, aus denen das MLP alle Linien berechnen muss. Das ist sehr wenig. Beim 2026-Modell (512×384) sind es **16×12×8 = 1536 Zahlen**. Das ist einer der Gründe, warum die Eingangsgröße ein so wichtiger Hebel ist (Kapitel 5).

---

## 3. Videos und Lesestoff

Alle Videos sind auf Englisch. Bei YouTube kann man unter ⚙️ → *Untertitel* → *Automatisch übersetzen* → *Deutsch* einstellen. Tipp: Auf 0,75× oder 1× Geschwindigkeit schauen und bei Formeln pausieren.

### Stufe 1: Wie lernt ein neuronales Netz? (Woche 1, ca. 2 Stunden)

Die beste Einführung, die es gibt, mit tollen Animationen.

| | Video | Länge | Worum geht's |
|---|---|---|---|
| 🎬 | [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | ~19 min | Neuronen, Schichten, Gewichte |
| 🎬 | [3Blue1Brown: Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | ~21 min | Loss und Gradientenabstieg = Schritte 3 und 5 der Schleife |
| 🎬 | [3Blue1Brown: Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | ~13 min | Schritt 4 der Schleife |
| 🎬 | [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) | ~10 min | *Optional:* die Mathematik dahinter |
| 🎬 | [StatQuest: A Gentle Introduction to Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI) | ~13 min | Trainings- vs. Testdaten, warum man überhaupt testet |

### Stufe 2: Wie sieht ein Netz Bilder? (Woche 1, ca. 1,5 Stunden)

Unser Backbone ist ein CNN (Convolutional Neural Network). Das muss ich verstehen, um Grad-CAM zu verstehen.

| | Video | Länge | Worum geht's |
|---|---|---|---|
| 🎬 | [3Blue1Brown: But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) | ~23 min | Was eine Faltung (Convolution) mathematisch macht |
| 🎬 | [StatQuest: Image Classification with CNNs](https://www.youtube.com/watch?v=HGwBXDKFk9I) | ~16 min | Filter, Feature Maps, Pooling, sehr anschaulich |
| 🎬 | [deeplizard: Convolutional Neural Networks explained](https://www.youtube.com/watch?v=YRhxdVk_sIs) | ~9 min | Kurze Zusammenfassung, gut zum Wiederholen |
| 🎬 | [DeepLearningAI: Transfer Learning](https://www.youtube.com/watch?v=FQM13HkEfBk) | ~8 min | Warum man mit vortrainierten Gewichten startet (unser `resnet18-…pth`) |

### Stufe 3: Wie trainiert man *gut*? (Woche 2, ca. 2 Stunden)

Das ist der wichtigste Block für meine Arbeit. Die Videos von Andrew Ng (DeepLearningAI, Kurs 2) sind kurz und erklären genau die Hyperparameter, die in unserer Config stehen.

| | Video | Länge | Passt zu Config-Parameter |
|---|---|---|---|
| 🎬 | [Train/Dev/Test Sets](https://www.youtube.com/watch?v=1waHlpKiNyY) | ~12 min | Validierung, `best_metric` |
| 🎬 | [Bias/Variance](https://www.youtube.com/watch?v=SjQyLhQIXSM) | ~8 min | Underfitting vs. Overfitting erkennen |
| 🎬 | [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) | ~6 min | Dasselbe nochmal anders erklärt |
| 🎬 | [Basic Recipe for Machine Learning](https://www.youtube.com/watch?v=C1N_PDHuJ6Q) | ~6 min | **Was tun bei welchem Problem**, sehr wichtig! |
| 🎬 | [Regularization](https://www.youtube.com/watch?v=6g0t3Phly2M) | ~10 min | `weight_decay` |
| 🎬 | [Mini Batch Gradient Descent](https://www.youtube.com/watch?v=4qJaSmvhxi8) | ~11 min | `batch_size` |
| 🎬 | [Adam Optimization Algorithm](https://www.youtube.com/watch?v=JXQT_vxqwIs) | ~7 min | `optimizer`, `momentum` |
| 🎬 | [Learning Rate Decay](https://www.youtube.com/watch?v=QzulmoOg2JE) | ~7 min | `scheduler`, `steps`, `gamma` |
| 🎬 | [Tuning Process](https://www.youtube.com/watch?v=AXDByU3D1hA) | ~7 min | Welche Hyperparameter sind am wichtigsten? |
| 🎬 | [Using an Appropriate Scale](https://www.youtube.com/watch?v=cSoK_6Rkbfg) | ~9 min | Warum Optuna die Learning Rate „log“ sucht |
| 🎬 | [Data Augmentation](https://www.youtube.com/watch?v=JI8saFjK84o) | ~10 min | `aug_*`-Parameter |

### Stufe 4: In das Modell hineinschauen (Woche 1–2)

| | Quelle | Worum geht's |
|---|---|---|
| 🎬 | [Grad-CAM: Demo-Video der Autoren](https://www.youtube.com/watch?v=COjUB9Izk6E) | Kurze Vorstellung von Grad-CAM |
| 📄 | [Grad-CAM-Paper (arXiv 1610.02391)](https://arxiv.org/abs/1610.02391) | Abschnitt 3 reicht. Erklärt die Formel |
| 📄 | [HiResCAM-Paper (arXiv 2011.08891)](https://arxiv.org/abs/2011.08891) | „Use HiResCAM instead of Grad-CAM“. Abstract + Abbildung 1 reichen |
| 📄 | [pytorch-grad-cam (GitHub)](https://github.com/jacobgil/pytorch-grad-cam) | Fertige Bibliothek mit Grad-CAM, HiResCAM und vielen weiteren Methoden |

### Für später (wenn ich tiefer einsteigen will)

| | Quelle | Worum geht's |
|---|---|---|
| 📄 | [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | **Pflichtlektüre!** Wie Profis an ein Training herangehen, Schritt für Schritt |
| 🎬 | [Karpathy: Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | ~2,5 h. Backpropagation selbst programmieren. Danach versteht man es wirklich |
| 🎬 | [freeCodeCamp: PyTorch for Deep Learning (Full Course)](https://www.youtube.com/watch?v=V_xro1bcAuA) | ~25 h, zum Nachschlagen. Kapitel 0–3 reichen für den Anfang |
| 📄 | [UFLDv2-Paper (arXiv 2206.07389)](https://arxiv.org/abs/2206.07389) | Unser Modell (schon gelesen ✅) |
| 📄 | [UFLDv2 Original-Code](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) | Basis für unseren Code |

---

## 4. Was soll ich trainieren? (Die richtige Reihenfolge)

Der häufigste Anfängerfehler: sofort an den Hyperparametern drehen. Profis machen es andersherum:

```
1. FEHLER VERSTEHEN      Wo genau ist das Modell schlecht?          → Aufgabe 1 (Fehlertabelle)
        │                Warum? Worauf schaut es?                   → Grad-CAM / HiResCAM (Kapitel 6)
        ▼
2. BASELINE              Das bestehende Training 1:1 nachmachen      → Experiment 0
        │                und denselben F1-Wert bekommen.
        │                Erst dann weiß ich, dass mein Setup stimmt.
        ▼
3. DATEN                 Fehlen Situationen im Datensatz?            → größter Hebel!
        │                (Reflexe, Kreuzungen, enge Kurven)
        │                Mehr / bessere Labels, passende Augmentation
        ▼
4. MODELL / AUFLÖSUNG    Größere Eingangsbilder, anderes Backbone    → mittlerer Hebel
        ▼
5. HYPERPARAMETER        Learning Rate, Batch Size, …                → Feinschliff
                         (Optuna hat das schon teilweise gemacht)
```

**Faustregel aus Andrew Ngs „Basic Recipe“:**

| Was ich sehe | Diagnose | Was hilft |
|---|---|---|
| Train-Loss hoch, Val-F1 schlecht | **Underfitting** (Modell zu schwach) | Größeres Modell / größere Eingangsbilder, länger trainieren, Learning Rate prüfen |
| Train-Loss sehr niedrig, Val-F1 deutlich schlechter | **Overfitting** (auswendig gelernt) | Mehr Daten, mehr Augmentation, mehr `weight_decay` |
| Beide gut, aber auf dem Auto schlecht | **Daten passen nicht zur Realität** | Neue Bilder vom echten Auto/der echten Strecke labeln |
| Loss springt wild / wird `nan` | **Learning Rate zu hoch** | Learning Rate ÷ 3 bis ÷ 10 |
| Loss sinkt kaum | **Learning Rate zu klein** oder Bug | Learning Rate × 3, Daten und Labels prüfen |

---

## 5. Hyperparameter: Was bedeutet was, und was passiert bei Änderungen?

Werte aus `models/best_model_bev_2026_config.py` (neues Modell) und `config/model_bev_config.py` (aktuell eingebundenes Modell).

⚠️ **Auffälligkeit 1:** In der 2026-Config stehen am Ende **zwei** angehängte Optuna-Blöcke (Trial 15 und Trial 23). Python überschreibt einfach, also gilt der **letzte** Block (Trial 23: `learning_rate = 0.00168`, `momentum = 0.935`, `weight_decay = 5.5e-6`). Das sollte ich mit dem Team klären: Welcher Trial war wirklich der beste?

⚠️ **Auffälligkeit 2:** Die 2026-Config sagt `backbone = "9"`, die Trainings-Config im Branch `origin/training` (`configs/smartrollerz_res18_bev.py`) sagt `backbone = "18"`. Prüfen, mit welchem Backbone `best_model_bev_2026.pth` wirklich trainiert wurde.

### 5.1 Die wichtigsten Trainings-Hyperparameter

| Parameter | Alt (bev) | 2026 | Was macht er? | Höher → | Niedriger → | Mein Startwert |
|---|---|---|---|---|---|---|
| `learning_rate` | 0.00625 (SGD) | 0.00168 (Adam) | Schrittgröße beim Lernen | schneller, aber instabil, Loss springt | stabil, aber langsam, bleibt evtl. hängen | 2026-Wert lassen. Wenn ändern, dann in Faktoren von 3 (z. B. 0.0005 / 0.0017 / 0.005) |
| `optimizer` | SGD | Adam | Wie die Schritte berechnet werden | – | – | **Adam** (verzeiht mehr, guter Standard) |
| `batch_size` | 64 | 32 | Bilder pro Schritt | glattere Gradienten, mehr GPU-Speicher, braucht oft höhere LR | verrauschter, oft bessere Generalisierung | 32 lassen |
| `epoch` | 250 | 250 | Wie oft alle Daten durchlaufen werden | länger, Risiko Overfitting | evtl. noch nicht fertig gelernt | 250. In der Loss-Kurve prüfen, ob es am Ende noch besser wird |
| `weight_decay` | 1e-4 | 5.5e-6 | L2-Regularisierung (bestraft große Gewichte) | weniger Overfitting, aber evtl. Underfitting | mehr Freiheit, mehr Overfitting-Risiko | Nur ändern, wenn ich Overfitting sehe → 1e-5 bis 1e-4 testen |
| `momentum` | 0.9 | 0.935 | Bei SGD: wie viel „Schwung“ mitgenommen wird | glatter, schießt evtl. übers Ziel | reagiert schneller, zittriger | Lassen. Bei Adam wird es laut Code ggf. gar nicht benutzt (prüfen!) |
| `scheduler`, `steps`, `gamma` | multi, [100,150], 0.1 | gleich | LR wird bei Epoche 100 und 150 jeweils ×0.1 kleiner | – | – | Lassen. Alternative später: `cos` (Cosine) |
| `warmup`, `warmup_iters` | linear, 100 | gleich | LR startet klein und steigt in den ersten 100 Schritten an | stabilerer Start | – | Lassen |

### 5.2 Modell- und Bild-Parameter (größter Hebel neben den Daten)

| Parameter | Alt (bev) | 2026 | Was macht er? | Effekt einer Änderung |
|---|---|---|---|---|
| `train_width` × `train_height` | 128×96 | 512×384 | Eingangsauflösung | **Größer:** feinere Details, genauere Linien, aber 16× mehr Rechenzeit pro Bild. Muss auf dem Auto noch schnell genug sein! Zwischenstufe 256×192 ist interessant |
| `backbone` | "9" | "9" (prüfen) | Größe des Feature-Extraktors | `18`/`34`: mehr Kapazität, langsamer. `9`: schnell |
| `num_row` / `num_col` | 35 / 35 | 55 / 40 | Anzahl Anker-Zeilen/-Spalten | Mehr Anker = mehr Punkte pro Linie, Kopf wird größer |
| `num_cell_row` / `num_cell_col` | 50 / 50 | 100 / 100 | In wie viele Zellen jede Anker-Zeile eingeteilt wird | Mehr Zellen = feinere Position, aber schwerer zu lernen |
| `mlp_mid_dim` | (Standard) | 512 | Breite der versteckten MLP-Schicht | Größer = mehr Kapazität im „Gehirn“ |
| `num_lanes` | 3 | 3 | links, Mitte, rechts | Nicht ändern |

⚠️ Wenn ich `train_width/height`, `num_row/col` oder `num_cell_*` ändere, ändert sich die **Form** des Netzes. Dann muss ich neu trainieren (kein Weitertrainieren mit `finetune`), und der ROS-Node braucht die passende Config.

### 5.3 Augmentation (nur im 2026-Modell, sehr wichtig gegen Overfitting)

Augmentation = Trainingsbilder bei jedem Durchlauf zufällig verändern, damit das Netz robuster wird.

| Parameter | Wert | Was passiert | Wann ändern? |
|---|---|---|---|
| `aug_translate_x` / `_y` | 150 / 30 px | Bild seitlich/vertikal verschieben | Mehr, wenn Linien am Rand verloren gehen |
| `aug_scale_min` / `_max` | 0.8 / 1.2 | Zoomen | – |
| `aug_rotate_deg` | 20° | Drehen | Mehr, wenn Kurven schlecht erkannt werden |
| `aug_brightness_min` / `_max` | 0.5 / 1.5 | Heller/dunkler | **Erweitern, wenn Lichtreflexe Geisterlinien erzeugen** |
| `aug_contrast_min` / `_max` | 0.5 / 1.5 | Kontrast | Wie Helligkeit |

**Ideen für neue Augmentations** (wenn die Fehlertabelle das zeigt):
- Künstliche Lichtflecken/Reflexe ins Bild malen (gegen Geisterlinien)
- Leichtes Rauschen oder Unschärfe (Kamera wackelt)
- Teile des Bildes zufällig abdecken („Cutout“), damit das Netz nicht nur auf eine Stelle schaut

### 5.4 Goldene Regeln beim Ändern

1. **Immer nur eine Sache auf einmal ändern.** Sonst weiß ich nicht, was geholfen hat.
2. **Immer gegen die Baseline vergleichen**, mit derselben Metrik (`val/local_f1`) und denselben Val-Bildern.
3. **Alles aufschreiben:** Config, Datum, Ergebnis. MLflow macht das automatisch, zusätzlich eine Zeile in die Tabelle in Kapitel 7.
4. **Zufall beachten:** Zwei Läufe mit gleicher Config können sich um ~0,5–1 % F1 unterscheiden. Ein Unterschied von 0,3 % ist also noch kein Beweis.
5. **Am Ende zählt das Auto.** F1 auf dem Val-Set ist nur ein Zwischenziel.

---

## 6. Grad-CAM und HiResCAM: Ins Modell hineinschauen

### 6.1 Was ist das?

Beide Methoden beantworten die Frage: **„Auf welche Stellen im Bild hat das Netz geschaut, um diese Entscheidung zu treffen?“** Das Ergebnis ist eine Heatmap über dem Bild: rot = wichtig, blau = egal.

So funktioniert es (vereinfacht):
1. Ich wähle eine Schicht im Backbone, z. B. `layer3`. Ihre Ausgabe sind viele „Feature Maps“ (Kanäle), jede ist ein kleines Bild.
2. Ich wähle eine Zahl, die ich erklären will, z. B. „wie sicher ist sich das Netz, dass die linke Linie *hier* liegt“ (der *Score*).
3. Backpropagation sagt mir für jede Feature Map und jede Position, wie stark der Score davon abhängt (der *Gradient*).
4. Feature Maps × Gradienten zusammenrechnen → Heatmap.

### 6.2 Der Unterschied zwischen Grad-CAM und HiResCAM

| | Grad-CAM | HiResCAM |
|---|---|---|
| Formel | Gradient pro Kanal **mitteln** → ein Gewicht pro Kanal → × Feature Map | Gradient **elementweise** × Feature Map (kein Mitteln) |
| Bedeutung | „Welche *Kanäle* sind wichtig?“, dann überall hinmalen, wo der Kanal aktiv ist | „Welche *Positionen* hat das Netz wirklich benutzt?“ |
| Problem | Kann Stellen markieren, die das Netz gar nicht benutzt hat (laut HiResCAM-Paper). Heatmaps oft größer und „schöner“ als die Wahrheit | Heatmaps oft fleckiger, aber ehrlicher |
| Wann gleich? | Bei Netzen mit Global Average Pooling am Ende (typische Klassifikationsnetze) sind beide gleich | |

**Warum HiResCAM für unser Modell besonders passt:** UFLDv2 hat **kein** Global Average Pooling. Der Kopf macht `flatten` und dann ein MLP. Das heißt: Für das Netz ist es extrem wichtig, **an welcher Position** ein Merkmal auftaucht. Genau daraus berechnet es ja, wo die Linie liegt! Grad-CAM wirft diese Positionsinformation beim Mitteln der Gradienten weg. HiResCAM behält sie. (Weil unser MLP zwei Schichten hat, ist auch HiResCAM nicht 100 % exakt, aber deutlich näher dran.)

⚠️ **Der Name täuscht:** „HiRes“ heißt nicht, dass die Heatmap eine höhere Auflösung hat. Die Auflösung hängt nur von der gewählten Schicht ab:

| Schicht | Auflösung bei 128×96 (altes Modell) | bei 512×384 (2026-Modell) | Was sie „sieht“ |
|---|---|---|---|
| `layer2` | 16×12 | 64×48 | Kanten, Strichstücke |
| `layer3` | 8×6 | 32×24 | Linienstücke, Muster |
| `layer4` | 4×3 | 16×12 | Ganze Strukturen (sehr grob!) |

Beim alten Modell ist `layer3` mit 8×6 schon sehr grob. Für feine Analysen lieber `layer2` nehmen oder das 2026-Modell untersuchen.

### 6.3 Unser Skript benutzen

Es gibt schon `scripts/gradcam_lanes.py` (offline, ohne ROS). Für jede Linie (links, Mitte, rechts) wird eine Heatmap mit den vorhergesagten Punkten gezeichnet.

🛠️ **Ausprobieren:**

```bash
cd ~/Documents/smarty_workspace/src/lane_detection_ai
~/.pyenv/versions/default/bin/python3 scripts/gradcam_lanes.py --layer layer3 --target loc
```

Die Bilder landen in `gradcam_output/`.

| Option | Werte | Bedeutung |
|---|---|---|
| `--target loc` | | „Warum liegt die Linie **hier**?“ |
| `--target exist` | | „Warum glaubt das Netz, dass es die Linie **gibt**?“ Gut für Geisterlinien! |
| `--layer` | `layer2` / `layer3` / `layer4` | Siehe Tabelle oben |
| `--frames` | z. B. `390,400,410` oder `all` | Welche Bilder aus dem Eval-Ordner |
| `--config` | Pfad zur Config | Für das 2026-Modell die 2026-Config mit `test_model` auf die `.pth` |

**HiResCAM einbauen:** Im Skript ist es eine Zeile Unterschied, in `GradCam.__call__`:

```python
# Grad-CAM (aktuell): Gradienten pro Kanal mitteln
weights = self.gradients.mean(dim=(2, 3), keepdim=True)
cam = torch.relu((weights * self.activations).sum(dim=1))[0]

# HiResCAM: Gradienten elementweise, ohne Mitteln
cam = torch.relu((self.gradients * self.activations).sum(dim=1))[0]
```

Am besten als neue Option `--method gradcam|hirescam`, damit ich beide nebeneinander vergleichen kann.

### 6.4 Wie lese ich die Heatmaps?

| Was ich sehe | Bedeutung | Was ich fürs Training daraus lerne |
|---|---|---|
| ✅ Heatmap liegt auf der Linie, die erklärt wird | Netz schaut auf das Richtige | – |
| ⚠️ Heatmap auf der **Nachbarlinie** | Netz leitet die Position aus der anderen Linie ab (z. B. „Mitte = links + Spurbreite“) | Nicht unbedingt schlimm, aber wenn die Nachbarlinie fehlt, versagt es → Bilder mit fehlenden Linien ins Training |
| ❌ Heatmap auf einem **Lichtreflex** | Reflex sieht für das Netz aus wie eine Linie | Mehr Trainingsbilder mit Reflexen, Helligkeits-Augmentation erweitern |
| ❌ Heatmap auf dem **Bildrand / schwarzen BEV-Bereich / Fahrzeugteil** | „Shortcut Learning“: Netz hat eine Abkürzung gelernt, die nichts mit Linien zu tun hat | Diese Bereiche maskieren/abschneiden, mehr Verschiebe-Augmentation (`aug_translate_x`) |
| ❌ `exist`-Heatmap bei einer Geisterlinie zeigt auf ein bestimmtes Objekt | Dieses Objekt löst die falsche Erkennung aus | Genau solche Bilder sammeln und labeln |
| ⚠️ Heatmap überall verschmiert | Schicht zu grob, oder das Netz ist unsicher | `layer2` probieren, HiResCAM probieren |

### 6.5 Vom Bild zur Verbesserung

Grad-CAM/HiResCAM trainieren nicht selbst, sie sind ein **Diagnose-Werkzeug**. Der Ablauf:

```
Fehlertabelle (Aufgabe 1)          z. B. „Frame 400: Geisterlinie rechts bei Reflex“
        │
        ▼
Grad-CAM/HiResCAM auf genau        --frames 395,400,405 --target exist
diese Frames                       → Heatmap liegt auf dem Reflex
        │
        ▼
Hypothese                          „Das Netz kennt zu wenige Reflexe“
        │
        ▼
Maßnahme                           mehr Reflex-Bilder labeln ODER Helligkeits-
                                   Augmentation erweitern
        │
        ▼
Neu trainieren, vergleichen        F1 besser? Heatmap jetzt auf der echten Linie?
```

**Vorher/Nachher-Vergleich:** Dieselben Frames mit dem alten und dem neuen Modell durch das Skript schicken und die Bilder nebeneinanderlegen. Das ist auch super für Präsentationen.

---

## 7. Experimentplan: Meine ersten Trainingsläufe

Jedes Experiment ändert **genau eine** Sache gegenüber der Baseline.

| # | Experiment | Änderung | Warum | Erwartung |
|---|---|---|---|---|
| 0 | **Baseline** | Nichts. 2026-Config 1:1 nachtrainieren | Prüfen, ob mein Setup funktioniert | Gleicher F1 wie das bestehende Modell (±1 %) |
| 1 | Augmentation Helligkeit | `aug_brightness` 0.3–1.8 | Lichtreflexe (falls Fehlertabelle das zeigt) | Weniger Geisterlinien |
| 2 | Auflösung | 256×192 statt 512×384 (Anker/Zellen anpassen!) | Schneller auf dem Auto | F1 etwas schlechter, aber deutlich schneller. Lohnt es sich? |
| 3 | Learning Rate | ×3 und ÷3 | Gefühl für die Empfindlichkeit bekommen | Vermutlich ist der Optuna-Wert schon gut |
| 4 | Mehr Daten | Eigene Fehlerfälle labeln (CVAT) und hinzufügen | Größter Hebel | Bessere Ergebnisse genau bei diesen Situationen |

**Ergebnistabelle:**

| Datum | # | Änderung | val/local_f1 | Train-Loss (Ende) | ms/Bild | Bemerkung / Grad-CAM-Eindruck |
|---|---|---|---|---|---|---|
| | 0 | Baseline | | | | |
| | | | | | | |
| | | | | | | |

---

## 8. Training auf deep-thought (DGX) starten

`deep-thought` (`141.31.112.54`) ist der GPU-Server: **8× NVIDIA H100 (je 80 GB)**, 224 CPU-Kerne, 2 TB RAM. Er wird von vielen geteilt und über **Slurm** verwaltet.

⚠️ **Niemals ein Training direkt im SSH-Terminal starten.** Immer über Slurm (`sbatch`/`srun`), sonst blockiere ich GPUs, die Slurm anderen zugeteilt hat.

**Vorlage im Branch `origin/training`:** `train_lightning.slurm`. Das Skript

- reserviert 1 GPU, 24 CPUs, 128 GB RAM für max. 8 Stunden (Partition `debug`)
- startet einen Enroot-Container (`.sqsh`) mit dem Trainingscode
- loggt die Ergebnisse nach MLflow

⚠️ Im Skript stehen **Pfade und Namen der Vorgänger** (`inf23174`, `USER="sanel"`, `/raid/homes/inf23174/jan/...`). Die muss ich auf meinen Account `inf24039` anpassen, und ich brauche eine eigene Kopie von Datensatz und Container. Vorher mit dem Team klären (Aufgabe 5)!

**Nützliche Slurm-Befehle:**

```bash
sbatch train_lightning.slurm      # Job abschicken
squeue -u $USER                   # Meine Jobs anzeigen
scancel <JOBID>                   # Job abbrechen
sinfo                             # Welche Partitionen / GPUs gibt es?
tail -f slurm-outs/slurm-<JOBID>.out   # Live-Ausgabe des Trainings
```

**Ergebnisse anschauen:** Mit `mlflow_ui.slurm` (auch im Branch `origin/training`) wird die MLflow-Oberfläche gestartet. Dort sehe ich die Loss- und F1-Kurven aller Läufe nebeneinander.

**Was ich in den Kurven prüfe:**
- Sinkt der Train-Loss? Wenn nicht: Learning Rate oder Daten falsch
- Steigt `val/local_f1`? Wenn er irgendwann wieder sinkt, während der Train-Loss weiter fällt: **Overfitting**
- Sieht man bei Epoche 100 und 150 einen Knick? Das ist der Scheduler (gewollt)

---

## 9. Checkliste

**Verstehen**
- ⬜ Stufe 1 Videos (neuronale Netze, Gradient Descent, Backpropagation)
- ⬜ Stufe 2 Videos (CNNs)
- ⬜ Stufe 3 Videos (Andrew Ng: Bias/Variance, Basic Recipe, Optimizer, Augmentation)
- ⬜ Karpathy „Recipe for Training Neural Networks“ gelesen
- ⬜ Kann in eigenen Worten erklären: Loss, Gradient, Learning Rate, Epoche, Overfitting

**Analysieren**
- ⬜ Fehlertabelle hat 5–10 Einträge (Aufgabe 1)
- ⬜ Grad-CAM auf die Fehler-Frames laufen lassen (`loc` und `exist`)
- ⬜ HiResCAM ins Skript eingebaut und mit Grad-CAM verglichen
- ⬜ Hypothesen notiert: Warum macht das Modell diese Fehler?

**Klären**
- ⬜ Welcher Optuna-Trial ist der richtige für die 2026-Config?
- ⬜ Backbone 9 oder 18 für das 2026-Modell?
- ⬜ DGX-Zugang, Datensatz, Container, Partition

**Trainieren**
- ⬜ Experiment 0 (Baseline) erfolgreich, F1 wie erwartet
- ⬜ Erstes eigenes Experiment mit genau einer Änderung
- ⬜ Ergebnis in Tabelle (Kapitel 7) und im Protokoll eingetragen
