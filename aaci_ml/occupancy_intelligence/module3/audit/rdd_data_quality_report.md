# AACI Module 3: RDD Dataset & Energy Semantics Technical Audit Report

## 1. Executive Summary & Inventory

The **Room-level Dataset (RDD)** consists of **675,332 raw telemetry records** collected from **9 smart ceiling fans** installed across Indian residential households, grouped into three distinct room types:
- **Bedrooms (3 fans):** `Bedroom_Fan_1` (79,384 rows), `Bedroom_Fan_2` (179,243 rows), `Bedroom_Fan_3` (110,409 rows).
- **Drawing Rooms (3 fans):** `Drawingroom_Fan_1` (60,444 rows), `Drawingroom_Fan_2` (43,529 rows), `Drawingroom_Fan_3` (77,764 rows).
- **Lounges (3 fans):** `Lounge_Fan_1` (21,305 rows), `Lounge_Fan_2` (55,488 rows), `Lounge_Fan_3` (47,766 rows).

Each record records: `S.No.`, `datetime`, `temperature`, `humidity`, `mode`, `speed`, `opTime`, `eSpent`, and `eSaved`.

---

## 2. Telemetry Packet Structure & Physical Energy Semantics

### 2.1 The Mathematical Definition of `opTime`, `eSpent`, and `eSaved`
Our code-level mathematical audit over all 675,332 records proved with **100.00% precision** that:
$$\text{eSpent} = \text{Power}(\text{speed}) \times \text{opTime}$$
$$\text{eSaved} = \text{SavedRate}(\text{speed}) \times \text{opTime}$$
$$\text{Baseline Induction Power}(\text{speed}) = \text{Power}(\text{speed}) + \text{SavedRate}(\text{speed})$$

| Fan Speed | State | BLDC Fan Power Rate ($P_{\text{BLDC}}$) | Modeled Baseline Induction Power ($P_{\text{Induction}}$) | Energy Saved Rate ($P_{\text{Saved}}$) | Ratio ($P_{\text{Saved}} / P_{\text{Induction}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | OFF | **0 W** | 0 W | 0 W | N/A |
| **1** | Ultra Low | **10 W** | 24 W | 14 W | 58.3% |
| **2** | Low | **15 W** | 34 W | 19 W | 55.9% |
| **3** | Medium | **20 W** | 55 W | 35 W | 63.6% |
| **4** | High | **27 W** | 64 W | 37 W | 57.8% |
| **5** | Maximum / Boost | **48 W** | 96 W | 48 W | 50.0% |

### 2.2 Physical Units & Conversion
- **`opTime`:** Operating time in **seconds** during the device reporting packet interval (modal value: 150 to 152 seconds $\approx 2.5$ minutes).
- **`eSpent`:** Energy consumed in **Joules (Watt-seconds)**:
  $$\text{Energy}_{\text{Wh}} = \frac{\text{eSpent}}{3,600}, \quad \text{Energy}_{\text{kWh}} = \frac{\text{eSpent}}{3,600,000}$$
- **`eSaved`:** Calculated energy saved in **Joules (Watt-seconds)** compared to a standard induction motor ceiling fan.

> [!IMPORTANT]
> **Measured vs. Estimated Clarification:**
> The RDD telemetry does **not** contain hardware current/voltage meter sensor measurements. The energy consumption `eSpent` is computed in firmware from operating duration and speed-calibrated power constants. Energy savings `eSaved` are synthetic modeled differences against nominal induction baseline wattages (24W to 96W).

---

## 3. Data Integrity & Preprocessing Audit Findings

### 3.1 Lexicographical vs. Chronological Ordering Bug
The raw CSV files were sorted alphabetically by the `datetime` string (`ORDER BY datetime` as VARCHAR). Because hours 1 through 9 lacked leading zeros (`1:00` vs `10:00`), text sorting placed `10:00`–`19:59` before `1:00`–`1:59` (e.g. `15:31` appeared before `1:00`).
- **Resolution:** All pipelines must parse timestamps using `dayfirst=True` (`DD/MM/YYYY HH:MM`) and explicitly sort by `(datetime, S.No.)`.

### 3.2 Duplicate Telemetry Packets
- **Total exact duplicate records:** 139,787 rows (20.70% of total dataset).
- `Bedroom_Fan_2` exhibited 129,802 duplicate rows (72.42%) due to network retries / server logging duplicates.
- **Resolution:** Deduplicate before session analysis and modeling.

### 3.3 Sensor Error Sentinels (0xFF / 255)
- Telemetry records with `temperature == 255` or `humidity == 255` represent standard embedded microcontroller `0xFF` timeout / bus error codes.
- `Lounge_Fan_3` had 558 error records; other fans had between 2 and 29 error records.
- **Resolution:** Filter out records where `temperature == 255` or `humidity == 255`.

---

## 4. Overall Dataset Audit Metrics Table

| Fan Identifier | Room Type | Raw Records | Duplicates | Dup % | Valid Span | Active Days | Total eSpent (kWh) | Total eSaved (kWh) | Formula Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bedroom_Fan_1** | Bedroom | 79,384 | 6,705 | 8.45% | 2021-03-05 to 2022-01-03 | 293 | 156.03 kWh | 184.28 kWh | **100.0%** |
| **Bedroom_Fan_2** | Bedroom | 179,243 | 129,802 | 72.42% | 2021-02-07 to 2022-04-15 | 393 | 662.23 kWh | 675.92 kWh | **100.0%** |
| **Bedroom_Fan_3** | Bedroom | 110,409 | 84 | 0.08% | 2020-11-10 to 2022-04-15 | 504 | 215.12 kWh | 239.58 kWh | **100.0%** |
| **Drawingroom_Fan_1** | Drawing Room | 60,444 | 46 | 0.08% | 2020-09-22 to 2022-03-09 | 499 | 236.00 kWh | 248.04 kWh | **100.0%** |
| **Drawingroom_Fan_2** | Drawing Room | 43,529 | 3,038 | 6.98% | 2021-01-13 to 2022-04-05 | 425 | 132.78 kWh | 146.43 kWh | **100.0%** |
| **Drawingroom_Fan_3** | Drawing Room | 77,764 | 25 | 0.03% | 2020-10-12 to 2021-09-10 | 332 | 56.88 kWh | 65.87 kWh | **100.0%** |
| **Lounge_Fan_1** | Lounge | 21,305 | 26 | 0.12% | 2020-09-28 to 2021-06-06 | 219 | 26.20 kWh | 30.85 kWh | **100.0%** |
| **Lounge_Fan_2** | Lounge | 55,488 | 20 | 0.04% | 2021-03-23 to 2022-03-18 | 338 | 247.88 kWh | 265.03 kWh | **100.0%** |
| **Lounge_Fan_3** | Lounge | 47,766 | 41 | 0.09% | 2021-01-11 to 2022-04-06 | 425 | 232.13 kWh | 244.73 kWh | **100.0%** |
| **TOTAL / OVERALL** | **All 9 Fans** | **675,332** | **139,787** | **20.70%** | **2020-09-22 to 2022-04-15** | **571 Span** | **1,965.25 kWh** | **2,100.74 kWh** | **100.0%** |

