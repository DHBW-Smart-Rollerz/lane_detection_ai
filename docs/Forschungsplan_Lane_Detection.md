# Forschungsplan: Lane Detection AI

> Stand: 31.10.2026 · Autorin: Brisild Capani
>
> Dieses Dokument hält fest, **was bisher untersucht wurde, mit welchem Ergebnis und warum**, und **wie es weitergeht**. Es ist so geschrieben, dass Abschnitte später direkt in die Arbeit übernommen werden können (Ausgangslage, Methodik, Ergebnisse).
> Begriffe: [Protokoll_Lane_Detection_AI.md](Protokoll_Lane_Detection_AI.md) · Aufgaben: [Aufgaben_Lane_Detection_AI.md](Aufgaben_Lane_Detection_AI.md) · Training: [Lernpfad_KI_Training.md](Lernpfad_KI_Training.md)
>
> **Hauptquelle** für die Ausgangslage ist die Studienarbeit der Vorgänger:
> J. Plieninger, S. Trnka: *Accelerating Edge Perception: Hardware-Aware Optimization of UFLDv2 on the Hailo-8 Accelerator for Autonomous Racing*, DHBW Stuttgart, 2026. Seitenangaben „(S. x)“ beziehen sich auf diese Arbeit.

---

## Inhaltsverzeichnis

1. [Ausgangslage](#1-ausgangslage)
2. [Beobachtete Schwächen des aktuellen Modells](#2-beobachtete-schwächen-des-aktuellen-modells)
3. [Bisherige Arbeiten und Ergebnisse](#3-bisherige-arbeiten-und-ergebnisse)
4. [Neue Ausrichtung: Frontansicht und alternative Backbones](#4-neue-ausrichtung-frontansicht-und-alternative-backbones)
5. [Forschungsfragen](#5-forschungsfragen)
6. [Messkonzept (Methodik)](#6-messkonzept-methodik)
7. [Nächste Schritte](#7-nächste-schritte)
8. [Offene Fragen ans Team](#8-offene-fragen-ans-team)
9. [Werkzeuge, Dateien und Orte](#9-werkzeuge-dateien-und-orte)

---

## 1. Ausgangslage

### 1.1 System

Die Spurerkennung des Fahrzeugs „Smarty“ ist ein ROS-2-Knoten (`lane_detection_ai_node`). Er erhält das Kamerabild nach Entzerrung und Vogelperspektiven-Transformation (Bird's-Eye-View, BEV) aus dem Knoten `camera_preprocessing`, erkennt mit einem neuronalen Netz bis zu drei Fahrbahnlinien (links, Mitte, rechts), rechnet die Punkte in Weltkoordinaten um und fittet pro Linie ein Polynom 2. Grades (S. 31–32).

### 1.2 Aktuelles Modell (Stand der Vorgänger)

| Eigenschaft | Wert | Quelle |
|---|---|---|
| Architektur | Ultra Fast Lane Detection v2 (UFLDv2), hybride Zeilen-/Spalten-Anker | S. 22–25 |
| Eingabe | BEV-Bild, 512 × 384 Pixel | Config |
| Backbone | ResNet-9, Klassifikationskopf `mlp_mid_dim = 512` | S. 41–43, 60 |
| Hardware im Fahrzeug | Hailo-8 (INT8, Post-Training-Quantisierung mit 1024 Kalibrierbildern) | S. 50–51 |
| Validierungs-F1 | 0,8972 | S. 57 |
| Latenz (Hailo-8) | 4,27 ms Inferenz, 7,76 ms gesamt | S. 60 |
| Latenz (CPU, gleiches Modell) | ca. 859 ms Inferenz | S. 60 |
| Fahrgeschwindigkeit | 1,0 m/s im Wettbewerb, 1,3 m/s im Labor; begrenzt durch die Lenkung, nicht durch die Wahrnehmung | S. 64, 69 |

Wichtige Entscheidungen der Vorgänger und ihre Begründung:

- **BEV statt Frontansicht:** Modelle auf der Frontansicht (Ego-Perspektive) waren bei ResNet-9/-18 deutlich schlechter. Die Lücke wurde ausdrücklich als **offene Frage für zukünftige Arbeiten** benannt (S. 38–39).
- **ResNet-9 statt ResNet-18:** ResNet-18 erreichte zwar F1 0,9452, war in Einzelfällen aber unruhiger und passt nicht in einen einzigen Hailo-Kontext (S. 42–43).
- **Keine MobileNetV3/YOLOv5s:** Beide trainierten gut (MobileNetV3 früh F1 0,8805), scheiterten aber beim ONNX-Export (u. a. HardSwish-Aktivierung). Das Debuggen wurde aus Zeitgründen ausgelassen (S. 40–41, Ausblick S. 70).

### 1.3 Einordnung

Die Vorgänger haben die **Geschwindigkeit** gelöst (≈ 100 ms → 7,76 ms). Offen geblieben sind:

1. **Robustheit:** Stabilität über die Zeit, Verdeckung, Lichtreflexe, Kurven. Die Nachverarbeitung nutzt einen einfachen `np.polyfit` ohne Ausreißerfilter und ohne zeitliche Glättung (S. 32).
2. **Frontansicht** als Eingabe (S. 39).
3. **Alternative Backbones**, die auf dem Hailo-8 laufen (S. 70).

---

## 2. Beobachtete Schwächen des aktuellen Modells

Beobachtet im Debug-Bild in RViz auf aufgezeichneten Fahrten (Okt. 2026):

| # | Situation | Beobachtung |
|---|---|---|
| S1 | **Licht/Reflex auf der Linie** | Die Punkte liegen nicht mehr genau auf der Linie, sondern schwanken |
| S2 | **Kurven / wechselnde Linienführung** | Die Linien schwanken stark. Die Punkte „wollen geradeaus“, obwohl die Linie abbiegt |
| S3 | **Verdeckte Linie** | Die Erkennung wird komplett instabil |

S2 deckt sich mit einem Fehlerbild, das schon die Vorgänger beschreiben: In Kurven wird die Mittellinie als Randlinie interpretiert und eine Linie dazwischen „erfunden“ (S. 48).

**Erste quantitative Messung** (Skript `scripts/measure_stability.py`, Fahrt `eval_2025-03-06`, 1192 Bilder, ungelabelt, 30.10.2026). „Sprung“ ist die Änderung der x-Position einer Linie zwischen zwei aufeinanderfolgenden Bildern, ausgewertet in den Bildzeilen 200/400/600:

| Linie | erkannt in | Median-Sprung | 95 %-Sprung | Sprünge > 30 px |
|---|---|---|---|---|
| links | 70,1 % | 3,0 px | 28,8 px | 4,8 % |
| Mitte | 97,3 % | 2,6 px | 21,1 px | 3,3 % |
| rechts | 99,7 % | 2,3 px | 15,7 px | 2,1 % |

Alle drei Linien gleichzeitig erkannt: **69,2 %** der Bilder.

---

## 3. Bisherige Arbeiten und Ergebnisse

### 3.1 Laufzeit auf der CPU (Laptop, 29.10.2026)

Ziel: Das 2026-Modell (512 × 384) lokal ausführbar machen, ohne Hailo.

| Variante | ms pro Bild (nur Netz) |
|---|---|
| PyTorch, 4 Threads | ≈ 800–1100 |
| **ONNX Runtime** (Standard-Threads) | **≈ 140–150** |

Die Ausgaben von PyTorch und ONNX waren auf 20 Testbildern pixelgenau identisch. Die Node nutzt lokal deshalb die ONNX-Datei (`config/model_bev_2026_config.py`). Ergebnisse von der CPU sind **nicht** auf den Hailo übertragbar. Die Werte decken sich mit den ≈ 859 ms der Vorgänger (S. 60).

### 3.2 Trainings-Pipeline auf der DGX (30.10.2026)

Das Training läuft auf der DGX H100 „deep-thought“ über Slurm mit Pyxis/Enroot-Containern. Docker steht für Studierenden-Accounts nicht zur Verfügung. Deshalb wird der Container mit Pyxis aus `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` gebaut und als `.sqsh` gespeichert.

Aufgetretene Probleme und Lösungen (für spätere Nutzer):

| Problem | Ursache | Lösung |
|---|---|---|
| `permission denied … docker.sock` | Account nicht in der `docker`-Gruppe | Pyxis statt Docker (`srun --container-image … --container-save …`) |
| `Read-only file system` bei `pip install` | Pyxis-Container sind standardmäßig schreibgeschützt | `--container-remap-root --container-writable` |
| Kaputter Container wurde trotzdem gespeichert | `--container-save` speichert auch nach Fehlern | `set -e` und bei Fehler `.sqsh` löschen |
| `MlflowException: filesystem tracking backend … maintenance mode` | MLflow ≥ 3.x verweigert den Datei-Speicher | `export MLFLOW_ALLOW_FILE_STORE=true` |
| `InvalidAccount` | vorübergehender Slurm-Zustand (Accounting ist abgeschaltet) | Job abbrechen und neu abschicken |
| Alte `train.slurm` scheitert | Datei der Vorgänger mit Docker und fehlerhafter Zeile `$PATH = …` | Eigene Datei `train_pyxis.slurm` |

Hinweis: Es werden jeweils die **neuesten** Paketversionen installiert (z. B. PyTorch Lightning 2.6, DALI 2.4, MLflow 3.17, OpenCV 5). Diese weichen von der Umgebung der Vorgänger ab.

### 3.3 Datensatz und Aufteilung

Verwendet wurde der lokal vorhandene, **ältere** Datenstand: `labels.zip` (CVAT-Exporte, `train_gt.txt`, Annotations-Cache) und `rosbags.zip` (Bilder, BEV-Bilder, Segmentierungsmasken).

- `train_gt.txt`: 2972 Einträge, 2544 verschiedene Bilder (manche Bilder sind mehrfach gelistet, also überabgetastet), aus Aufnahmen von 2023 und 05/2025.
- Zum Vergleich der Datensatz der Vorgänger: 4647 Trainings- und 342 Validierungsbilder (S. 49). Es fehlen u. a. 608 Bilder mit Polfilter und 723 schwierige Kurven.
- Ein Validierungs-Set war nicht vorhanden. Deshalb wurden **vier komplette Aufnahmesessions zurückgehalten** (418 Einträge, 14 %): `2023-05-24-17-55-35`, `2023-06-28-15-20-08`, `2023-12-19-15-11-20`, `rosbag2_2025_05_08-18_45_32`. Die Aufteilung erfolgte nach Session, damit fast identische Nachbarbilder nicht gleichzeitig in Training und Validierung landen.

### 3.4 Baseline-Training (ResNet-9, BEV)

Konfiguration: `configs/smartrollerz_res18_bev.py` mit `backbone = "9"` und `mlp_mid_dim = 512`. Adam, Lernrate 3,8·10⁻⁴, Batch 32, 250 Epochen, Lernrate ×0,1 bei Epoche 100 und 150, keine Early-Stopping-Regel. Dauer ≈ 40 min auf einer H100.

| Epoche | Validierungs-F1 |
|---|---|
| 9 | 0,72 |
| 19 | 0,82 |
| 47 | 0,85 |
| 99 → 100 (Lernrate ↓) | 0,877 → 0,904 |
| Bestwert (Epoche 228) | 0,920 |
| letzte 20 Epochen | Ø 0,903 (0,892–0,913) |

Der Validierungs-Loss fiel bis zum Ende mit dem Trainings-Loss mit (2,80 bzw. 2,66), es gab also keine Anzeichen von Overfitting.

### 3.5 Vergleich mit dem 2026-Modell

**Test 1 – F1 auf den 418 zurückgehaltenen Bildern** (je 5 Messungen, Mittelwert ± Standardabweichung):

| | 2026-Modell | Baseline (neu) |
|---|---|---|
| F1 | 0,952 ± 0,004 | 0,907 ± 0,005 |
| Precision | 0,961 | 0,914 |
| Recall | 0,943 | 0,900 |

⚠️ **Eingeschränkt aussagekräftig:** Die Vorgänger haben alle älteren Bilder ins Training übernommen (S. 48). Das 2026-Modell hat die zurückgehaltenen Sessions daher sehr wahrscheinlich **im Training gesehen** (Datenleck).

**Test 2 – Stabilität auf einer Fahrt, die keines der Modelle kennt** (`eval_2025-03-06`):

| | 2026-Modell | Baseline (neu) |
|---|---|---|
| linke Linie erkannt | 70,1 % | 78,7 % |
| linke Linie, Sprünge > 30 px | 4,8 % | 6,6 % |
| Mitte / rechts | nahezu gleich | nahezu gleich |
| alle 3 Linien erkannt | 69,2 % | 78,7 % |

**Ergebnis:** Die Baseline ist nicht besser als das 2026-Modell. Das war zu erwarten, weil sie mit weniger Daten aus demselben Bestand trainiert wurde. Der Lauf bestätigt aber, dass die Trainings-Pipeline vollständig funktioniert.

### 3.6 Methodische Erkenntnisse

1. **Die Validierung im Trainingscode ist nicht deterministisch.** `get_eval_loader` nutzt denselben DALI-Lader wie das Training, **mit zufälliger Augmentierung** (Verschiebung, Rotation, Helligkeit). Derselbe Checkpoint ergibt deshalb bei jeder Messung einen anderen F1 (Schwankung ≈ ±0,005). Der im Training gespeicherte „Bestwert“ ist optimistisch (0,920 im Training gegenüber 0,907 ± 0,005 bei wiederholter Messung).
2. **Es gibt kein unabhängiges Test-Set.** Auch die Vorgänger haben auf dem Set gemessen, mit dem sie das beste Modell ausgewählt haben (S. 49).
3. **Der F1 misst keine Stabilität über die Zeit.** Ein Modell kann einen hohen F1 haben und trotzdem zittern. Deshalb braucht es zusätzlich die Sprung-Messung.

---

## 4. Neue Ausrichtung: Frontansicht und alternative Backbones

Empfehlung des Vorgängers (Gespräch, Okt. 2026): **Auf die Frontansicht konzentrieren** und **andere Backbones als ResNet** (z. B. YOLO) testen, die der Hailo-Compiler verarbeiten kann.

### 4.1 Motivation

| Argument | Beleg |
|---|---|
| BEV sieht nur ca. **1 m** voraus, die Frontansicht bis zum Horizont, also mehr Vorausschau für Kurven | S. 39 |
| BEV hängt stark an der **Kalibrierung**. Die Kamerahalterung ist locker, deshalb wird jede Aufnahme neu kalibriert. Verrutscht die Kamera, entsteht eine Domänenverschiebung | S. 33, 69 |
| Die Lücke zwischen Frontansicht und BEV wurde nur mit ResNet-9/-18 untersucht und ausdrücklich offen gelassen | S. 39 |
| Effizientere Backbones (MobileNetV3, YOLOv5s) waren vielversprechend, scheiterten nur am Export | S. 40, 70 |
| **Mögliche Laufzeitersparnis:** Ohne BEV muss nicht das ganze Bild verzerrt werden. Es reicht, die wenigen erkannten Punkte in Weltkoordinaten umzurechnen | Hypothese, zu messen |

### 4.2 Was sich bei der Frontansicht ändert

- **Daten:** Die CVAT-Labels wurden auf **entzerrten Frontbildern** erstellt und erst danach in BEV umgerechnet (S. 34–35, `convert_cvat_bev.py` mit Schachbrettbild). In `rosbags.zip` liegen zu jedem Bild das Frontbild (`…_frame000110.png`, 5162 Dateien), das BEV-Bild und die Maske. Mit `scripts/convert_cvat.py` (Branch `training`) lassen sich die Labels für die Frontansicht erzeugen. Eine Config dafür existiert: `configs/smartrollerz_res18.py` (`data_root = "dataset/smartrollerz"`).
- **Geometrie:** Linien laufen in der Frontansicht zum Fluchtpunkt zusammen, und Kurven verlaufen flacher durchs Bild. Die **Spalten-Anker** von UFLDv2 sind hier wichtiger als in BEV (S. 22–23, 38).
- **Nachverarbeitung:** `bird_to_world` muss durch eine Umrechnung Frontbild → Welt ersetzt werden (Homographie auf die Punkte statt aufs Bild).
- **Label-Format:** TUSimple speichert pro Zeile nur einen x-Wert. Waagerechte Linien (90°-Kreuzung) lassen sich nicht darstellen (S. 35–36). In der Frontansicht betrifft das mehr Situationen, also prüfen.

### 4.3 Backbone-Kandidaten

Auswahlkriterium: Das Netz muss sich nach ONNX exportieren **und** mit dem Hailo Dataflow Compiler (DFC) kompilieren lassen. Im Branch `training` sind bereits eingebaut: ResNet-9/18/34/50/…, `mobilenet-v3-small`, `mobilenet-v3-large`, `yolov5n`, `yolov5s` (`model/backbone.py`, Configs `smartrollerz_mobileV3.py`, `smartrollerz_YOLOv5.py`).

| Kandidat | Stand | Zu klären |
|---|---|---|
| ResNet-9 | Referenz, läuft auf dem Hailo | — |
| ResNet-18 | F1 0,9452 in BEV, braucht mehrere Hailo-Kontexte | Latenz und Anzahl Kontexte mit kleinerem Kopf |
| MobileNetV3-small/-large | im Code, F1 0,8805 früh (S. 40) | ONNX-Export mit HardSwish reparieren (z. B. neueres Opset oder HardSwish durch eine Hailo-taugliche Form ersetzen) |
| YOLOv5n/s (Backbone) | im Code | Export und Kompilierung testen |
| Weitere aus dem Hailo Model Zoo (z. B. EfficientNet-Lite, RegNet) | noch nicht im Code | nur Architekturen wählen, für die Hailo eine kompilierbare Referenz anbietet |

Bewertet wird jeder Kandidat **in BEV und in der Frontansicht** nach: F1 (faires Test-Set), Stabilität, Latenz auf dem Hailo, Anzahl Hailo-Kontexte und F1-Verlust durch INT8.

### 4.4 Weitere Ideen

- **Synthetische Verdeckung** (gegen S3): Im Training Linienstücke abdecken, das Label aber vollständig behalten. Das Modell lernt so, verdeckte Linien aus dem Kontext zu schließen, was der globale Klassifikationskopf grundsätzlich kann (S. 24).
- **Lokale Licht-Augmentierung** (gegen S1): Lichtflecken und Reflexe statt nur globaler Helligkeit/Kontrast 0,5–1,5 (S. 46).
- **Robuste Nachverarbeitung** (gegen S1–S3): Mindestanzahl Punkte, RANSAC, Fallback über die Spurbreite, zeitliche Glättung (exponentielles Mittel oder Kalman-Filter).
- **Kleinerer Kopf:** Die letzte Kopfschicht (`cls.3`, 512 × 29070) hat ≈ 15 Mio. der ≈ 20 Mio. Parameter. Weniger Gitterzellen verkleinern sie stark, was auf dem Hailo zählt (Speicher, Kontexte).
- **Distillation:** ResNet-18 als Lehrer für ResNet-9.

---

## 5. Forschungsfragen

*(Entwurf, mit Betreuung abstimmen)*

**Hauptfrage**
> Lässt sich mit der Frontansicht und Hailo-kompatiblen alternativen Backbones eine Spurerkennung erreichen, die robuster ist als das bisherige BEV-ResNet-9-Modell, bei gleichbleibender Echtzeitfähigkeit auf dem Hailo-8 (< 10 ms)?

**Teilfragen**
1. **Eingabe:** Wie unterscheiden sich BEV und Frontansicht in Genauigkeit, Stabilität und Gesamtlatenz, wenn dasselbe Backbone verwendet wird?
2. **Backbone:** Welche Hailo-kompatiblen Backbones (ResNet, MobileNetV3, YOLOv5, …) bieten den besten Kompromiss aus F1, Stabilität und Latenz? Wie groß ist jeweils der Verlust durch INT8-Quantisierung?
3. **Robustheit:** Wie stark verbessern synthetische Verdeckung und lokale Licht-Augmentierung die Erkennung bei verdeckten Linien und Reflexen?
4. **Stabilität:** Wie stark reduzieren robuster Fit und zeitliche Glättung das Wackeln, und welche Latenz kosten sie?

---

## 6. Messkonzept (Methodik)

### 6.1 Grundregeln

1. **Unabhängiges Test-Set:** Gelabelte Bilder, die in **keinem** Training vorkommen, auch nicht im 2026-Modell. Am besten neue Aufnahmen oder `eval_*`-Fahrten, in CVAT gelabelt.
2. **Deterministische Auswertung:** Augmentierung im Eval-Lader abschalten. Bis dahin jede Messung **5-mal** wiederholen und `Mittelwert ± Standardabweichung` angeben.
3. **Gleiche Bilder, gleiches Skript** für jedes Modell. Für jede Zahl festhalten: Datum, Modell, Config, Test-Set, Hardware.
4. **Eine Änderung pro Experiment.**

### 6.2 Test-Sets und Kennzahlen pro Schwäche

| Schwäche | Test-Set | Kennzahl |
|---|---|---|
| S1 Licht | 50–100 gelabelte Bilder mit Reflexen | F1, Precision (Geisterlinien), seitliche Abweichung in px/cm |
| S2 Kurven | 50–100 gelabelte Kurvenbilder | F1, Abweichung im oberen Bilddrittel |
| S3 Verdeckung | gelabelte Testbilder mit **künstlich** abgedeckten Linien (0/25/50/75 %), Label unverändert | F1 je Verdeckungsgrad (Kurve) |
| Wackeln | ungelabelte Fahrt(en), z. B. `eval_2025-03-06` | `scripts/measure_stability.py`: Erkennungsrate, Median-/95 %-Sprung, Anteil Sprünge > 30 px |
| Gesamt | unabhängiges Test-Set | F1, Precision, Recall |

### 6.3 Laufzeit messen

| Ebene | Werkzeug | Aussage | Hardware |
|---|---|---|---|
| Node, ganze Pipeline | Timer-Ausgabe der Node (`inference`, `pred2coords`, `total`), `scripts/benchmark_models.py` | ms pro Stufe | Laptop (CPU) oder NUC (Hailo) |
| Nur Netz auf dem Hailo | `hailortcli benchmark <modell>.hef` | FPS, Latenz | Hailo-8 |
| Schätzung ohne Hardware | `hailo profiler <modell>.har` (DFC) | geschätzte Latenz/FPS, **Anzahl Kontexte** | keine |
| INT8-Genauigkeit ohne Hardware | DFC-Emulator (`ClientRunner`, Modus `SDK_QUANTIZED`) | F1 des quantisierten Modells | keine |

**Hailo testen ohne Fahrt:** NUC mit Hailo-Karte (oder das aufgebockte Fahrzeug) starten und die Eval-Bilder mit `scripts/image_folder_publisher.py` einspielen. So haben auch die Vorgänger gemessen (S. 58).

### 6.4 Ergebnistabelle (Vorlage für die Arbeit)

| Modell | Eingabe | F1 (Test) | F1 Verdeckung 50 % | Precision Licht | Sprünge > 30 px | Hailo ms | Kontexte |
|---|---|---|---|---|---|---|---|
| ResNet-9 (2026) | BEV | | | | 4,8 % (links) | 4,27 (S. 60) | 1 |
| ResNet-9 | Front | | | | | | |
| MobileNetV3 | Front | | | | | | |
| YOLOv5s | Front | | | | | | |

---

## 7. Nächste Schritte

| Prio | Schritt | Status |
|---|---|---|
| 1 | Kompletten Datensatz der Vorgänger holen (4989 Bilder inkl. 342 Sonnenlicht-Validierungsbilder) | 🔄 Download läuft |
| 2 | Bei Sanel erfragen: finale Config (AdamW + Cosine?), DFC-Container, Kalibrierbilder | ⬜ |
| 3 | Validierung deterministisch machen (Augmentierung im Eval-Lader aus) | ⬜ |
| 4 | Unabhängiges Test-Set festlegen und in CVAT labeln (Licht, Kurven, normal) | ⬜ |
| 5 | Verdeckungs-Test-Set automatisch aus dem Test-Set erzeugen (Skript) | ⬜ |
| 6 | Ausgangswerte des 2026-Modells auf allen Test-Sets messen („Vorher“) | ⬜ |
| 7 | Frontansicht-Datensatz erzeugen (`convert_cvat.py`) | ⬜ |
| 8 | ResNet-9 **Front** gegen ResNet-9 **BEV** trainieren und vergleichen (Teilfrage 1) | ⬜ |
| 9 | MobileNetV3 / YOLOv5: ONNX-Export reparieren, mit DFC kompilieren, Kontexte prüfen | ⬜ |
| 10 | Backbone-Vergleich (Teilfrage 2), jeweils mit INT8-Emulator | ⬜ |
| 11 | Verdeckungs- und Licht-Augmentierung (Teilfrage 3) | ⬜ |
| 12 | Robuster Fit und Glättung in der Node (Teilfrage 4) | ⬜ |
| 13 | Bestes Modell auf dem Hailo im NUC messen | ⬜ |

---

## 8. Offene Fragen ans Team

- Wo liegt der vollständige Datensatz, und welcher Stand war die Grundlage für das 2026-Modell?
- Welche Config (Optimizer, Lernrate, Scheduler, Early Stopping) hatte das finale Modell?
- Welche Bilder waren im Training des 2026-Modells? (wichtig, um ein wirklich unabhängiges Test-Set zu wählen)
- Wo liegen DFC-Container und Kalibrierbilder für `scripts/compile_pth_to_hef.slurm`?
- Welche Paketversionen hatte die Trainingsumgebung (`pip freeze`)?
- Gibt es Labels für die Frontansicht direkt, oder nur über `convert_cvat.py`?
- Woran ist der ONNX-Export von MobileNetV3/YOLOv5s genau gescheitert (Fehlermeldung)?

---

## 9. Werkzeuge, Dateien und Orte

### 9.1 Im Repo (`src/lane_detection_ai`)

| Datei | Zweck |
|---|---|
| `scripts/measure_stability.py` | Stabilität (Sprünge, Erkennungsrate) mehrerer Modelle auf einer Bildfolge |
| `scripts/image_folder_publisher.py` | Bildordner als ROS-Topic abspielen (Test ohne Kamera) |
| `scripts/benchmark_models.py` | Laufzeit mehrerer Configs in der Node vergleichen |
| `scripts/convert_pth_onnx.py` | `.pth` → `.onnx` (Normalisierung im Graphen) |
| `scripts/compile_pth_to_hef.slurm`, `scripts/compile_onnx_hef.sh` | Kompilierung nach `.hef` (braucht DFC) |
| `config/model_bev_2026_config.py` | 2026-Modell (lokal als ONNX) |
| `config/model_baseline_res9_20261030_config.py` | Baseline vom 30.10.2026 (ONNX) |

`config/*.py` und `models/**` stehen in `.gitignore` und existieren nur lokal.

### 9.2 Auf der DGX (`~/brisild/lane_detection_ai_training`, Klon von Branch `training`)

| Datei | Zweck |
|---|---|
| `train_pyxis.slurm` | Baut den Container einmalig, danach Training (ResNet-9-BEV-Config) |
| `unpack_dataset.slurm` | Entpackt `labels.zip`, `data.zip`, `rosbags.zip` nach `dataset/smartrollerz_bev` und setzt die Aufteilung |
| `eval_weights.py`, `eval_models.slurm` | F1 mehrerer Gewichtsdateien auf demselben Validierungs-Set (5 Wiederholungen) |
| `configs/smartrollerz_res9_bev.py` | ResNet-18-BEV-Config mit `backbone = "9"`, `mlp_mid_dim = 512` |
| `dataset/split/` | `train_gt.txt` / `eval_gt.txt` der Session-Aufteilung, Original als `train_gt_full.txt` |
| `results/<datum>_lr_…/best_weights.pth` | bestes Modell eines Laufs |
| `results/mlruns/` | MLflow-Daten (Kurven aller Läufe) |
| `~/brisild/containers/lane_detection_ai_training.sqsh` | Trainings-Container (14 GB) |

⚠️ Diese Dateien sind **nicht versioniert**, und die DGX wird am Semesterende aufgeräumt. Skripte daher in einen eigenen Branch übernehmen und Ergebnisse regelmäßig herunterladen.

### 9.3 Nützliche Befehle

```bash
# DGX: eigene Jobs, Log live, Job abbrechen
squeue -u inf24039
tail -F ~/brisild/lane_detection_ai_training/slurm-<jobid>.out
scancel <jobid>

# DGX: bester F1 des neuesten Laufs
cd $(dirname $(dirname $(dirname $(ls -t ~/brisild/lane_detection_ai_training/results/mlruns/*/*/metrics/val/local_f1 | head -1))))/metrics
awk '$3<250' val/local_f1 | sort -g -k2 | tail -1

# Laptop: Datei zur DGX hochladen / Modell herunterladen
rsync -avP <datei> deep-thought:~/brisild/lane_detection_ai_training/dataset/
rsync -avP deep-thought:~/brisild/lane_detection_ai_training/results/<lauf>/best_weights.pth models/

# Laptop: Stabilität vergleichen
cd ~/Documents/smarty_workspace/src/lane_detection_ai
PYTHONPATH=. python3 scripts/measure_stability.py config/model_bev_2026_config.py config/<anderes>_config.py
```
