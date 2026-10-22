# Aufgaben: Lane Detection AI

> Meine Aufgabenliste mit Erklärungen und Chronologie.
> Begriffe und Hintergründe stehen in [Protokoll_Lane_Detection_AI.md](Protokoll_Lane_Detection_AI.md).

**Legende:** ✅ erledigt · 🔄 in Arbeit · ⬜ offen

---

## Teil 1: Chronologie (was bisher passiert ist)

### 20.10.2026: Einarbeitung

- ✅ Workspace analysiert: ROS-2-Pipeline Kamera → `camera_preprocessing` → `lane_detection_ai` → `pathplanning`
- ✅ Modell verstanden: UFLDv2 mit ResNet-9-Backbone, BEV-Bilder 128×96, 3 Linien
- ✅ Erste Auffälligkeiten gefunden:
  - Das 2026-Modell ist nicht eingebunden
  - `.hef` (Hailo) wird nicht unterstützt
  - Bug `image.shape[0] != 3`
  - Lokale Änderung in `ros_params.yaml`
  - Trainingscode fehlt im Repo

### 21.10.2026: Begriffe und Protokoll

- ✅ Begriffe der Vorgänger aufgearbeitet: Kalibrierung, RViz2, DGX, Lightning, TensorBoard, Optuna, Kalman-Filter, L2-Regularisierung, Overfitting, F1-Score, Backbones, Hyperparameter, Optimizer
- ✅ Protokoll `docs/Protokoll_Lane_Detection_AI.md` erstellt, auf Branch `brisild` committet und gepusht

### 22.10.2026: Paper, Code, erster Testlauf

- ✅ Remote-Branches entdeckt:
  - `origin/training`: Trainingscode, Slurm-Skripte für die DGX, Dockerfile, CVAT-Konvertierung, Benchmark-Skript
  - `origin/hailo-8-accelerator`: Hailo-Unterstützung
- ✅ UFLDv2-Paper gelesen und zusammengefasst (arXiv 2206.07389)
- ✅ Code Schritt für Schritt verstanden, vom Bild bis zum publizierten Ergebnis
- ✅ Node gestartet: Kalibrierung vom 19.10.2026 und Modell laden fehlerfrei
- ✅ Offline-Test ohne Kamera eingerichtet:
  - `scripts/image_folder_publisher.py` spielt die 1192 BEV-Bilder aus `models/eval_2025-03-06/` auf `/camera/image/bev` ab
  - `config/lane_debug.rviz` ist das fertige RViz-Layout mit Debug-Bild und BEV-Bild
- ✅ **Erste Messung** (Laptop, PyTorch, `best_model_bev.pth`, Debug an):

  | Stufe | ms/Bild |
  |---|---|
  | `inference` | ~8 |
  | `pred2coords` | ~6,5 ⚠️ |
  | `debug_image` | ~2 |
  | Rest | ~2 |
  | **`total`** | **~20** (ca. 50 FPS möglich) |

  → `pred2coords` ist fast so teuer wie das Netz selbst. Laut Paper sollte der Head unter 5 % der Zeit brauchen.

---

## Teil 2: Testumgebung (so starte ich alles)

Ich brauche 3 Terminals, in jedem läuft ein Befehl:

**Terminal 1:** Lane-Node mit Debug-Bild
```bash
ros2 launch lane_detection_ai lane_detection_ai.launch.py debug:=true
```

**Terminal 2:** Bilder abspielen. Mit `--rate 2` geht es langsam zum Anschauen, mit `--loop` in Endlosschleife.
```bash
~/.pyenv/versions/default/bin/python3 ~/Documents/smarty_workspace/src/lane_detection_ai/scripts/image_folder_publisher.py --rate 2
```

**Terminal 3:** RViz mit fertigem Layout
```bash
rviz2 -d ~/Documents/smarty_workspace/src/lane_detection_ai/config/lane_debug.rviz
```

**Prüfen, ob alles läuft:**
```bash
ros2 node list
```
Dort müssen `/image_folder_publisher` und `/lane_detection_ai_node` stehen.
```bash
ros2 topic hz /lane_detection/lane
```
Die Rate sollte ungefähr der `--rate` des Publishers entsprechen.

**Farben im Debug-Bild:** 🔵 links · 🟢 Mitte · 🔴 rechts · Cyan = Trajektorien

---

## Teil 3: Meine Aufgaben (in dieser Reihenfolge)

### Aufgabe 1: Fehler des Modells anschauen und notieren ⬜

**Aufwand:** ca. 1 Stunde

**Warum:** Bevor ich etwas verbessere, muss ich wissen, *wo* das Modell schlecht ist. Sonst optimiere ich ins Blaue.

**Was tun:**
1. Testumgebung starten (Teil 2), Publisher mit `--rate 2`
2. In RViz auf diese Situationen achten:
   - Kurven: Folgen die Punkte der Linie?
   - Helle Stellen und Lichtreflexe (z. B. ab Frame ~400): Entstehen Geisterlinien?
   - Kreuzungen und Haltelinien
   - Fehlt eine Linie? Dann fehlt die Trajektorie auf der Seite
3. Jeden Fehler in die Fehlertabelle unten eintragen. Die Bildnummer steht ungefähr im Publisher-Log.

**Fertig, wenn:** Die Fehlertabelle hat mindestens 5–10 Einträge und ich weiß, welche Fehlerart am häufigsten ist.

---

### Aufgabe 2: Bugfix `image.shape[0]` + Git-Ablauf üben ⬜

**Aufwand:** ca. 10–30 Minuten

**Warum:** Ein kleiner, ungefährlicher Fix, um den Ablauf **ändern → testen → committen → pushen** einmal selbst zu machen.

**Was ist falsch:** In `lane_detection_ai/model/model_wrapper.py`, Funktion `predict()`:
```python
if image.shape[0] != 3:            # prüft die BILDHÖHE, nicht die Kanäle!
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
```
Das funktioniert nur zufällig, weil das Bild immer Graustufen ist. Bei einem Farbbild würde es abstürzen.

**Fix:**
```python
if image.ndim == 2:                # 2 Dimensionen = Graustufen (H, W)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
```

**Ablauf:**
1. Datei ändern
2. Neu bauen:
   ```bash
   cd ~/Documents/smarty_workspace && colcon build --symlink-install --packages-select lane_detection_ai
   ```
3. Testen: Node starten, Bilder abspielen, in RViz prüfen, ob alles gleich aussieht
4. Committen und pushen:
   ```bash
   cd ~/Documents/smarty_workspace/src/lane_detection_ai
   git add lane_detection_ai/model/model_wrapper.py
   git commit -m "Fix grayscale check in model wrapper"
   git push origin brisild
   ```

**Fertig, wenn:** Der Fix liegt auf GitHub im Branch `brisild`.

---

### Aufgabe 3: `pred2coords` schneller machen (vektorisieren) ⬜

**Aufwand:** 1–2 Tage

**Warum:** Die Funktion braucht ~6,5 ms von ~20 ms pro Bild, also ein Drittel der Gesamtzeit.

**Was ist das Problem:** Die Funktion läuft in einer **Python-Schleife** über jeden Anker (35 Zeilen × 3 Linien + 35 Spalten × 3 Linien = 210 Durchläufe). In jedem Durchlauf stehen viele kleine Torch-Aufrufe (`.item()`, `torch.where`, `softmax`). Jeder einzelne Aufruf hat Overhead.

**Idee:** Alle Anker **auf einmal** mit Matrix-Operationen berechnen (vektorisieren):
1. `argmax` für alle Anker gleichzeitig
2. Die Nachbarzellen (±1) per Indexing für alle gleichzeitig holen
3. `softmax` und Erwartungswert für alle gleichzeitig
4. Erst ganz am Ende mit der Existenz-Maske filtern und in Punktlisten umwandeln

**Wichtig, das Ergebnis muss gleich bleiben:**
- Ein Vergleichstest prüft, ob alte und neue Funktion auf allen 1192 Bildern **dieselben Punkte** liefern
- Erst dann die alte Version ersetzen

**Fertig, wenn:**
- Ergebnis identisch (Vergleichstest bestanden)
- `pred2coords` liegt bei ≤ 1 ms (Messung vorher/nachher ins Protokoll)
- Committet auf `brisild`

---

### Aufgabe 4: ONNX gegen PyTorch vergleichen ⬜

**Aufwand:** ½ Tag

**Warum:** Dasselbe Modell liegt auch als `models/best_model_bev.onnx` vor. ONNX Runtime ist auf der CPU oft schneller.

**Was tun:**
1. Prüfen, ob die ONNX-Config richtig in `config/` liegt (die Datei `model_onnx_bev_config.py` ist bisher nur in `models/`)
2. In `config/ros_params.yaml` `model_config_path` umstellen und neu bauen
3. Den `inference`-Wert vergleichen und in RViz prüfen, ob die Erkennung gleich gut ist
4. Ergebnis in die Vergleichstabelle im Protokoll (Kapitel 7) eintragen

**Achtung:** `config/ros_params.yaml` hat schon eine Änderung von jemand anderem. Vorher mit dem Team klären, welche Config auf dem Auto laufen soll.

**Fertig, wenn:** Tabelle mit ms/Bild für `.pth` und `.onnx` ausgefüllt, Empfehlung notiert.

---

### Aufgabe 5: Offene Fragen ans Team klären ⬜ (parallel zu allem)

**Warum:** Ohne diese Antworten kann ich nicht trainieren und nicht auf dem Auto testen.

- ⬜ Darf ich den Branch `origin/training` nutzen? Wer kennt sich damit aus?
- ⬜ **DGX-Zugang:** Adresse, Account, wie starte ich Slurm-Jobs?
- ⬜ Wo liegen der **Datensatz** (`smartrollerz_bev`) und die **Rosbags** (NAS)?
- ⬜ Mit welchem Tool wird gelabelt? (Vermutlich CVAT, siehe `convert_cvat_bev.slurm`)
- ⬜ Wann kann ich **am echten Auto** testen? Auf welchem Rechner ist die Kamera (NUC)?
- ⬜ Welche `ros_params.yaml` soll gelten, `.pth` oder `.onnx`?
- ⬜ Ist eine **Hailo-Karte** verbaut?
- ⬜ Welche **Val-F1-Werte** haben die bisherigen Modelle?

---

### Spätere Aufgaben (nach 1–5) ⬜

| Aufgabe | Warum | Aufwand |
|---|---|---|
| **Fallback bei fehlender Linie** | Fehlt Mitte oder Rand, ist die Trajektorie `[0,0,0]`. Besser wäre: die vorhandene Linie um eine feste Spurbreite verschieben | ½ Tag |
| **Ausreißer filtern vor `polyfit`** | Einzelne falsche Punkte verbiegen das Polynom (z. B. mit RANSAC) | ½ Tag |
| **Echte Hybrid-Anker** | Aktuell werden Zeilen- und Spalten-Punkte gemischt. Laut Paper sollte jede Linie nur *einen* Anker-Typ nutzen | 1 Tag |
| **Kalman-Filter / Glättung** | Linien zittern von Bild zu Bild. Glättung über die Zeit | 1–2 Tage |
| **RViz-Marker** | Linien zusätzlich als 3D-Marker in Metern anzeigen, nicht nur im Bild | ½ Tag |
| **2026-Modell einbinden** | Größeres Modell (512×384, Optuna-optimiert). Anker-Formel prüfen! | 1–2 Tage |
| **Hailo-Support (`.hef`)** | Inferenz auf dem KI-Beschleuniger, Branch `hailo-8-accelerator` anschauen | 2–3 Tage |
| **Neu trainieren** | Mit eigenen Fehlerfällen als zusätzlichen Trainingsdaten auf der DGX | offen |
| **Test am echten Auto** | Am Ende zählt, wie das Auto fährt | laufend |

---

## Teil 4: Fehlertabelle (aus Aufgabe 1)

| # | Bild / Frame | Situation | Was ist falsch | Linie | Idee |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Situation:** Gerade / Kurve / Kreuzung / Lichtreflex / Haltelinie / …
**Was ist falsch:** Linie fehlt / Geisterlinie / Punkte springen / falsche Linie / Trajektorie falsch

---

## Teil 5: Messwerte (vorher / nachher)

| Datum | Änderung | Rechner | `inference` ms | `pred2coords` ms | `total` ms | Bemerkung |
|---|---|---|---|---|---|---|
| 22.10.2026 | Ausgangszustand (`.pth`) | Laptop | ~8 | ~6,5 | ~20 | Debug an |
| | | | | | | |
| | | | | | | |
