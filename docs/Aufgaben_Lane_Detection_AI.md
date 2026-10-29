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

### 26.10.2026: Grad-CAM, Lernpfad, Update auf v0.2.0

- ✅ `scripts/gradcam_lanes.py` geschrieben: zeigt pro Linie, auf welche Bildbereiche das Netz schaut (`--target loc` / `exist`)
- ✅ Lernpfad `docs/Lernpfad_KI_Training.md` erstellt: Videos, Hyperparameter, Grad-CAM/HiResCAM, Experimentplan
- ✅ DGX gefunden: `deep-thought` (8× H100, Slurm, Enroot). Vorlage `train_lightning.slurm` im Branch `origin/training`
- ✅ **Sanels Hailo-8-Merge (PR #7, v0.2.0)** in `brisild` gemergt. Damit sind erledigt:
  - Bug `image.shape[0] != 3` → jetzt `image.ndim == 2` (war Aufgabe 2)
  - `pred2coords` vektorisiert → neues `model/decode.py` (war Aufgabe 3, Messung steht noch aus)
  - `.hef`-Support (Hailo-8) für das Auto
  - Neues Benchmark-Skript `scripts/benchmark_models.py`, Konvertierungs- und Kompilier-Skripte
- ✅ Lokale `config/model_bev_config.py`: `mlp_mid_dim = 512` ergänzt (neuer Standard ist 2048, alte Modelle haben 512)
- ✅ `gradcam_lanes.py` an die neue API angepasst (`IMAGE_MEAN/STD`, `load_checkpoint`, `decode_lanes`/`merge_lanes`)
- ✅ Erste Grad-CAM-Beobachtung: Bei Frame 400 liegt die Heatmap der **Mittellinie** auf dem Kreuz-Symbol in der Kurve, nicht auf der Linie selbst

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

### ✅ Erledigt durch Sanels Merge (v0.2.0)

| Alte Aufgabe | Was passiert ist |
|---|---|
| ~~Aufgabe 2: Bugfix `image.shape[0]`~~ | In `model_wrapper.py` steht jetzt `if image.ndim == 2:`. Den Git-Ablauf habe ich mit dem Grad-CAM-Commit trotzdem geübt |
| ~~Aufgabe 3: `pred2coords` vektorisieren~~ | Neues `model/decode.py`: ein numpy-Decoder ohne Python-Schleife über die Anker, für alle Backends. Nur die **Messung** fehlt noch → Aufgabe 2 |
| ~~Hailo-Support (`.hef`)~~ | `model/hailo.py`, Backend wird über die Dateiendung gewählt |

---

### Aufgabe 1: Fehler des Modells anschauen und notieren 🔄

**Aufwand:** ca. 1–2 Stunden

**Warum:** Bevor ich etwas verbessere, muss ich wissen, *wo* das Modell schlecht ist. Sonst optimiere ich ins Blaue.

**Was tun:**
1. Testumgebung starten (Teil 2), Publisher mit `--rate 2`
2. In RViz auf diese Situationen achten:
   - Kurven: Folgen die Punkte der Linie?
   - Helle Stellen und Lichtreflexe (z. B. ab Frame ~400): Entstehen Geisterlinien?
   - Kreuzungen, Haltelinien, Symbole auf der Fahrbahn (Kreuz bei Frame 400!)
   - Fehlt eine Linie? Dann fehlt die Trajektorie auf der Seite
3. Jeden Fehler in die Fehlertabelle (Teil 4) eintragen. Die Bildnummer steht ungefähr im Publisher-Log.
4. Für jeden Fehler-Frame Grad-CAM laufen lassen (siehe Aufgabe 3) und den Eindruck in die Spalte „Idee“ schreiben.

**Fertig, wenn:** Die Fehlertabelle hat mindestens 5–10 Einträge und ich weiß, welche Fehlerart am häufigsten ist.

---

### Aufgabe 2: Neue Laufzeiten messen (`.pth` / `.onnx`) ⬜

**Aufwand:** ca. 1–2 Stunden

**Warum:** Ich will wissen, was der Umbau gebracht hat. Vorher brauchte `pred2coords` ~6,5 ms von ~20 ms.

**Was tun:**
1. Neues Skript anschauen: `scripts/benchmark_models.py` (nutzt eine temporäre Params-Datei, `ros_params.yaml` bleibt unverändert)
2. `.pth` und `.onnx` messen. Für `.pth` braucht die Config `mlp_mid_dim = 512` (schon ergänzt)
3. In RViz prüfen, ob die Erkennung bei beiden gleich aussieht
4. Werte in Teil 5 eintragen

**Fertig, wenn:** Tabelle in Teil 5 hat Werte für `.pth` und `.onnx` nach dem Update, und ich weiß, ob `pred2coords` jetzt ≤ 1 ms ist.

---

### Aufgabe 3: Grad-CAM / HiResCAM für die Fehleranalyse ⬜

**Aufwand:** ½ Tag

**Warum:** Grad-CAM zeigt, *warum* das Modell einen Fehler macht (worauf es schaut). Daraus leite ich ab, welche Trainingsdaten fehlen. Details: [Lernpfad, Kapitel 6](Lernpfad_KI_Training.md#6-grad-cam-und-hirescam-ins-modell-hineinschauen).

**Was tun:**
1. Skript auf die Fehler-Frames aus Aufgabe 1 laufen lassen:
   ```bash
   cd ~/Documents/smarty_workspace/src/lane_detection_ai
   PYTHONPATH=. ~/.pyenv/versions/default/bin/python3 scripts/gradcam_lanes.py --frames 395,400,405 --target loc
   PYTHONPATH=. ~/.pyenv/versions/default/bin/python3 scripts/gradcam_lanes.py --frames 395,400,405 --target exist --layer layer2
   ```
2. HiResCAM als Option `--method gradcam|hirescam` einbauen (eine Zeile Unterschied, siehe Lernpfad 6.3)
3. Grad-CAM und HiResCAM nebeneinander vergleichen
4. Frame 400 genauer anschauen: Schaut die Mittellinie wirklich auf das Kreuz-Symbol?

**Fertig, wenn:** Zu jedem Fehler in der Fehlertabelle gibt es eine Vermutung, *warum* er passiert. HiResCAM-Option ist committet.

---

### Aufgabe 4: Fallback, wenn eine Linie fehlt ⬜

**Aufwand:** ½ Tag

**Warum:** In `lane_detection_ai_node.py` (Abschnitt `polyfit`) wird eine Fahrspur nur gebaut, wenn **beide** Nachbarlinien erkannt wurden. Sonst bleibt sie `[0, 0, 0]`. Fehlt die Mittellinie kurz, hat das Auto für **beide** Spuren keine Trajektorie.

**Idee:**
- Linke + Mitte da → wie bisher Mittelwert
- Nur eine der beiden da → diese Linie um die halbe Spurbreite verschieben. Nach `bird_to_world` sind die Punkte in Metern, das Polynom ist `x = a·y² + b·y + c`, also reicht es, `c` zu verschieben
- Spurbreite als ROS-Parameter in `ros_params.yaml` (vorher am Auto/in der Simulation messen)

**Fertig, wenn:** Bei fehlender Mittellinie gibt es trotzdem plausible Trajektorien (in RViz geprüft), committet auf `brisild`.

---

### Aufgabe 5: Robusterer Polyfit ⬜

**Aufwand:** ½ Tag

**Warum:** `np.polyfit` läuft auch bei ganz wenigen Punkten. Ein einzelner falscher Punkt (z. B. von einem Reflex) verbiegt das ganze Polynom.

**Was tun:**
1. Mindestanzahl an Punkten verlangen (z. B. ≥ 5), sonst Linie als „nicht erkannt“ behandeln
2. Ausreißer filtern, z. B. mit RANSAC oder einem zweiten Fit ohne die Punkte mit dem größten Fehler
3. Mit den Fehler-Frames aus Aufgabe 1 vorher/nachher vergleichen

**Fertig, wenn:** Einzelne Ausreißer verbiegen die Linie sichtbar weniger, committet auf `brisild`.

---

### Aufgabe 6: Offene Fragen ans Team klären 🔄 (parallel zu allem)

**Warum:** Ohne diese Antworten kann ich nicht trainieren und nicht auf dem Auto testen.

- ⬜ Darf ich den Branch `origin/training` nutzen? Wer kennt sich damit aus?
- 🔄 **DGX-Zugang:** Server ist `deep-thought`, mein Account `inf24039` funktioniert. Offen: Welche Partition darf ich nutzen? Wo liegen Container (`.sqsh`) und Datensatz? (Das Slurm-Skript enthält noch Pfade von `inf23174`/`sanel`)
- ⬜ Wo liegen der **Datensatz** (`smartrollerz_bev`) und die **Rosbags** (NAS)?
- ⬜ Mit welchem Tool wird gelabelt? (Vermutlich CVAT, siehe `convert_cvat_bev.slurm`)
- ⬜ Wann kann ich **am echten Auto** testen? Auf welchem Rechner ist die Kamera (NUC)?
- ⬜ Welche `ros_params.yaml` soll gelten: `.pth`, `.onnx` oder `.hef`?
- ⬜ Ist eine **Hailo-8-Karte im Auto** verbaut? (In meinem Rechner ist keine.)
- ⬜ Welche **Val-F1-Werte** haben die bisherigen Modelle?
- ⬜ **2026-Modell:** Backbone `9` oder `18`? In der Config stehen zwei Optuna-Blöcke (Trial 15 und 23), welcher ist der richtige?

---

### Spätere Aufgaben (nach 1–6) ⬜

| Aufgabe | Warum | Aufwand |
|---|---|---|
| **`.hef` gegen `.pth` vergleichen** | Durch die Quantisierung (int8) kann die Genauigkeit sinken. Dieselben Eval-Bilder mit beiden auswerten und Punkte vergleichen | ½ Tag |
| **Glättung über die Zeit** | Linien zittern von Bild zu Bild. Erst einfaches exponentielles Mittel der Koeffizienten, später Kalman-Filter | 1–2 Tage |
| **Echte Hybrid-Anker** | `merge_lanes` in `decode.py` mischt noch Zeilen- und Spalten-Punkte. Laut Paper sollte jede Linie nur *einen* Anker-Typ nutzen | 1 Tag |
| **RViz-Marker** | Linien zusätzlich als 3D-Marker in Metern anzeigen, nicht nur im Bild | ½ Tag |
| **2026-Modell einbinden** | Größeres Modell (512×384, Optuna-optimiert, mit Augmentation). `mlp_mid_dim = 512` ist schon in der Config. Anker-Formel prüfen! | 1–2 Tage |
| **Neu trainieren** | Mit eigenen Fehlerfällen als zusätzlichen Trainingsdaten auf `deep-thought`. Plan: [Lernpfad, Kapitel 7](Lernpfad_KI_Training.md#7-experimentplan-meine-ersten-trainingsläufe) | offen |
| **Test am echten Auto** | Am Ende zählt, wie das Auto fährt | laufend |

---

## Teil 4: Fehlertabelle (aus Aufgabe 1)

| # | Bild / Frame | Situation | Was ist falsch | Linie | Idee |
|---|---|---|---|---|---|
| 1 | 400 | Kurve mit Kreuz-Symbol | Grad-CAM: Heatmap der Mittellinie liegt auf dem Kreuz, nicht auf der Linie (in RViz noch prüfen) | Mitte | Mehr Bilder mit Fahrbahn-Symbolen? Mit HiResCAM gegenprüfen |
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
| | Nach v0.2.0 (`.pth`, neues `decode.py`) | Laptop | | | | Aufgabe 2 |
| | Nach v0.2.0 (`.onnx`) | Laptop | | | | Aufgabe 2 |
| | | | | | | |
