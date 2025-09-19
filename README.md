# In-Trigger

Computer-vision trigger stack for behavioral assays. Detect motion in user-defined ROIs, fire an Arduino output with fixed lag/on-time, and record synchronized video + TSV logs. When you use these codes, please reference this repository.

---

## Modules

### In-Trigger Core (Python)

Detects motion, filters by size/aspect, confirms hits across frames, maps ROI hits to scheduled trigger ON/OFF events, and writes MP4 + tab-delimited logs.

**Inputs**

* Live camera frames
* ROI definitions
* Size bounds and confirmation frames
* Trigger lag and on-time

**Outputs**

* `{timestamp}.mp4` track video
* `{timestamp}.txt` TSV log
* `{timestamp}.jpg` snapshot

### In-Trigger GUI (PySide6)

End-to-end control panel: camera select, ROI editor, detection params, serial/trigger timing, output directory, record control.

### In-Trigger Arduino (Sketch)

Minimal serial I/O endpoint. Parses newline-terminated commands from the PC and switches digital pins accordingly.

---

## Quick Start

### Requirements

* Python 3.10+
* Packages: `PySide6`, `opencv-python`, `numpy`, `pyserial`
* OS: Windows verified

### Run

Place all three files in one folder:

```
In_Trigger_Core.py
In_Trigger_GUI.py
In_Trigger_Arduino.ino
```

Then:

```bash
python In_Trigger_GUI.py
```

---

## Usage

1. **Camera**: pick device.
2. **ROIs**: draw rectangles or circles; center dots + IDs render in preview.
3. **Detect**: set min/max area and confirmation frames; tune with filtered view.
4. **Serial/Trigger**: select COM, set **Lag (s)** and **On (s)**.
5. **Record**: choose output folder and start. App writes MP4 + TSV + snapshot.

**Serial protocol**

* ROI `i` maps to Arduino digital pin `i+1`.
* Commands from PC (newline-terminated):

  * `N<pin>` → set HIGH
  * `F<pin>` → set LOW

---

## File Outputs

* **Video**: `YYMMDD_HHMMSS.mp4`
* **Snapshot**: `YYMMDD_HHMMSS.jpg`
* **Log (TSV)**: `YYMMDD_HHMMSS.txt` with columns:
  `Date	Time	ROI_1_Objects … ROI_N_Objects	Trigger_Status	Video_Name	Frame_Number`

Log events include `Object Detected`, `Shock ON`, `Shock OFF`.

---

## Contact

Author: **Babur Erdem**
Email: **[ebabur@metu.edu.tr](mailto:ebabur@metu.edu.tr)**

