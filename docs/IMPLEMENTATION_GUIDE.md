# Sign Language Translate Implementation Guide

## Goal

สร้างแอปแปลภาษามือเป็นข้อความ โดยเริ่มจาก MVP ที่เล็กและวัดผลได้ก่อน:

1. เปิดกล้อง
2. ตรวจจับมือ
3. แปลงมือเป็น landmark features
4. ทำนาย gesture จากชุดคำจำกัด
5. แสดงข้อความที่แปลได้แบบ realtime

ข้อควรจำ: ภาษามือไม่ใช่ universal language ต้องระบุก่อนว่าจะทำ ASL, Thai Sign Language หรือ custom gesture set สำหรับ demo

## Recommended Roadmap

### Phase 1: Python ML Prototype

Stack:

- Python
- OpenCV
- MediaPipe
- scikit-learn
- pandas / numpy
- joblib

เหตุผล:

- ทดลองเร็วที่สุด
- debug dataset และ model ง่าย
- ยังไม่ต้องแบกรับความซับซ้อนของ mobile/native ML
- เหมาะกับการพิสูจน์ว่า recognition pipeline ใช้ได้จริง

Current files:

- `phase1_sign_recognition/collect_dataset.py`: เก็บ landmark dataset จาก webcam
     Dataset                      ภาษา                  ประเภท                    เหมาะกับ
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TSL-51                       Thai Sign Language    videos + landmarks        เหมาะสุดถ้าจะทำภาษาไทย
  ───────────────────────────  ────────────────────  ────────────────────────  ──────────────────────────────────
   ASL Alphabet                 ASL                   static images A-Z         เหมาะกับ MVP ง่าย ๆ
  ───────────────────────────  ────────────────────  ────────────────────────  ──────────────────────────────────
   WLASL                        ASL                   word-level videos         เหมาะกับ recognition ระดับคำ
  ───────────────────────────  ────────────────────  ────────────────────────  ──────────────────────────────────
   MS-ASL                       ASL                   large-scale videos        เหมาะกับ research/model ใหญ่
  ───────────────────────────  ────────────────────  ────────────────────────  ──────────────────────────────────
   ASL Citizen                  ASL                   84k videos, 2.7k signs    dataset ใหญ่และ realistic
  ───────────────────────────  ────────────────────  ────────────────────────  ──────────────────────────────────
   NECTEC Thai Sign Language    Thai                  video + annotation        เหมาะกับงานวิจัย แต่ format อาจซับซ้อน

  ตัวที่ผมแนะนำให้ลองก่อน

  1. TSL-51: Thai Sign Language Dataset
      - มี video, landmark data, metadata
      - มี 51 Thai signs + null_act
      - เหมาะกับโปรเจกต์เรา เพราะ Phase 1 ใช้ landmark อยู่แล้ว
      - License: CC BY-NC-SA 4.0, ใช้เชิง research/non-commercial
      - https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51

  2. ASL Alphabet Dataset
      - ง่ายสุดสำหรับเริ่ม classification
      - เป็นภาพ static A-Z
      - ไม่ตรงกับ Thai Sign Language แต่เหมาะสำหรับทดสอบ pipeline
      - https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset

  3. WLASL
      - ASL word-level video 2,000 words
      - เหมาะถ้าอยากขยับจาก alphabet ไปเป็น “คำ”
      - แต่ license จำกัด academic/computational use และไม่ใช่ commercial
      - https://dxli94.github.io/WLASL/

  4. MS-ASL
      - ASL dataset ขนาดใหญ่ มี subsets เช่น MS-ASL100, 200, 500, 10000
      - เหมาะกับงาน research มากกว่า MVP
      - https://www.microsoft.com/en-us/download/details.aspx?id=100121

  5. ASL Citizen
      - ประมาณ 84k videos, 2.7k ASL signs
      - realistic กว่า เพราะเป็น community-sourced
      - เหมาะกับ isolated sign recognition
      - https://www.microsoft.com/en-us/research/?p=932478
- `phase1_sign_recognition/train_model.py`: train classifier จาก CSV
- `phase1_sign_recognition/predict_realtime.py`: ทำนาย gesture realtime
- `phase1_sign_recognition/sign_translate_phase1/landmarks.py`: extract และ normalize landmarks
- `data/raw/`: dataset CSV local
- `models/`: trained model local

### Phase 2: Web Demo

Stack ที่แนะนำ:

- React
- Vite
- TypeScript
- MediaPipe Tasks Vision for Web
- Optional: FastAPI ถ้าจะเรียก model จาก backend

เหมาะสำหรับ demo/portfolio เพราะ deploy ง่ายและเปิดกล้องจาก browser ได้

### Phase 3: Backend API

Stack:

- FastAPI
- Python model inference
- SQLite ตอนเริ่ม
- PostgreSQL ถ้าต้องเก็บ users, sessions, feedback, model versions

หน้าที่ backend:

- รับ landmark sequence
- ส่ง prediction กลับ
- เก็บ prediction logs
- version model

### Phase 4: Mobile App

ตัวเลือก:

- React Native + Expo ถ้าต้องการ cross-platform และเคยใช้ React
- Flutter + TFLite ถ้าอยากได้ mobile UI ดีและเน้น on-device inference
- Native Android Kotlin + MediaPipe ถ้าเน้น Android performance

แนะนำให้ทำ mobile หลังจาก model และ preprocessing ใน Phase 1 นิ่งแล้ว

## Stack Comparison Summary

| Stack | เหมาะกับ | จุดเด่น | จุดอ่อน |
|---|---|---|---|
| React Web + MediaPipe + Python ML | Demo และ portfolio | ทำเร็ว deploy ง่าย | ยังไม่ใช่ native mobile |
| React Native Expo + TFLite/FastAPI | Mobile app | codebase เดียวสำหรับ iOS/Android | native camera/ML ซับซ้อนขึ้น |
| Flutter + TFLite | Mobile performance | UI ดี on-device ML ได้ | learning curve และ integration สูงกว่า web |
| Kotlin + MediaPipe | Android จริงจัง | performance และ native APIs ดี | ทำได้เฉพาะ Android |
| Python + OpenCV + MediaPipe | Research/prototype | train/test model ง่ายสุด | ไม่ใช่ user-facing app |

## Phase 1 Workflow

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Dataset

เก็บอย่างน้อย 2 labels ก่อน เช่น `hello`, `thanks`

```bash
python3 phase1_sign_recognition/collect_dataset.py --label hello --samples 120
python3 phase1_sign_recognition/collect_dataset.py --label thanks --samples 120
```

Controls:

- `SPACE`: start/pause capture
- `q`: quit

คำแนะนำตอนเก็บข้อมูล:

- เก็บ label ละ 100-300 samples ก่อน
- เปลี่ยนมุมมือ ระยะกล้อง และแสงเล็กน้อย
- อย่าเพิ่งเก็บ gesture ที่คล้ายกันมากจน model แยกยาก
- เริ่มจาก static gestures ก่อน เช่น hello, yes, no, thanks

### 3. Train Model

```bash
python3 phase1_sign_recognition/train_model.py
```

ผลลัพธ์:

- แสดง classification report
- save model ไปที่ `models/sign_classifier.joblib`

## Using TSL-51 Dataset

ถ้า clone TSL-51 มาแล้ว ให้ชี้ path ไปที่ root ของ dataset เช่นโฟลเดอร์ที่มี `landmarks`, `metadata`, และ `videos`

Inspect format:

```bash
python3 phase1_sign_recognition/inspect_tsl51.py /path/to/thai-sign-language-tsl51
```

Convert เป็น CSV format ของ Phase 1:

```bash
python3 phase1_sign_recognition/convert_tsl51_landmarks.py /path/to/thai-sign-language-tsl51 --output data/raw/tsl51_landmarks.csv
```

หมายเหตุ: converter จะ normalize landmark features ให้ตรงกับ webcam pipeline และจะไม่โหลด `.pkl` / `.pickle` โดย default. ถ้า dataset local เชื่อถือได้และจำเป็นต้องอ่าน pickle ให้เพิ่ม `--allow-unsafe-pickle`.

สำหรับ CSV ของ TSL-51 converter จะใช้ label จาก `metadata/user_sign_metadata.csv`
และเก็บ `source_id` ต่อ frame เพื่อให้ training แยก train/validation ตาม recording
แทนการสุ่มแยก frame จาก video เดียวกัน. ปัจจุบันยังใช้ one-hand model ดังนั้น
two-hand signs จะยังไม่ถูกแปลอย่างถูกต้อง.

Train ด้วยข้อมูล TSL-51:

```bash
python3 phase1_sign_recognition/train_model.py --input data/raw/tsl51_landmarks.csv --model models/tsl51_sign_classifier.joblib
```

Quick conversion test แบบจำกัด rows:

```bash
python3 phase1_sign_recognition/convert_tsl51_landmarks.py /path/to/thai-sign-language-tsl51 --output /private/tmp/tsl51_sample.csv --max-rows 1000
python3 phase1_sign_recognition/train_model.py --input /private/tmp/tsl51_sample.csv --model /private/tmp/tsl51_sample_model.joblib --estimators 50
```

### 4. Realtime Prediction

```bash
python3 phase1_sign_recognition/predict_realtime.py
```

Controls:

- `q`: quit

ถ้า predict สั่นหรือมั่ว:

- เพิ่ม samples
- ลดจำนวน labels ก่อน
- เก็บข้อมูลในสภาพแสงที่หลากหลายขึ้น
- เพิ่ม confidence threshold เช่น `--threshold 0.7`

## Technical Notes

### Landmark Features

MediaPipe คืนค่า hand landmarks 21 จุด แต่ละจุดมี `x`, `y`, `z`

ใน Phase 1 เราใช้:

```text
21 points * 3 values = 63 features per frame
```

Normalization:

- ใช้ wrist เป็น origin
- scale ด้วยระยะ landmark ที่ไกลสุดจาก wrist
- ทำให้ model ทนต่อขนาดมือและระยะกล้องมากขึ้น

### Model Choice

Phase 1 ใช้ `RandomForestClassifier`

เหตุผล:

- train ง่าย
- ไม่ต้องใช้ GPU
- ใช้กับ tabular landmark features ได้ดีพอสำหรับ MVP
- อ่านผลลัพธ์และ debug ง่ายกว่า neural network

ถ้าต้องทำ gesture แบบเคลื่อนไหวต่อเนื่อง ให้เปลี่ยนจาก single-frame classifier เป็น sequence model:

- LSTM
- GRU
- Temporal CNN
- Transformer

Input จะเปลี่ยนจาก:

```text
[63]
```

เป็น:

```text
[frames, 63]
```

## Known Limitations

- รองรับมือเดียวเป็นหลัก
- เหมาะกับ static gestures มากกว่า sentence-level sign language
- ยังไม่มี grammar/context handling
- ยังไม่รองรับ two-hand signs อย่างจริงจัง
- ยังไม่มี temporal smoothing
- dataset เป็นของ local machine และถูก ignore จาก git

## Next Improvements

1. เพิ่ม temporal smoothing เช่น majority vote 10-15 frames ล่าสุด
2. รองรับ two-hand landmarks โดยใช้ 126 features
3. เพิ่ม script สำหรับ clean/balance dataset
4. เพิ่ม confusion matrix เพื่อดูว่า label ไหนสับสนกัน
5. เพิ่ม Web UI ด้วย React + Vite
6. ทำ FastAPI inference endpoint
7. export model เป็น TFLite หรือ ExecuTorch เมื่อจะทำ on-device mobile

## Project Rule

ตอนพัฒนาใน Codex environment นี้ shell command ควร prefix ด้วย:

```bash
rtk <command>
```

เช่น:

```bash
rtk git status
rtk pytest -q
rtk npm run build
```
