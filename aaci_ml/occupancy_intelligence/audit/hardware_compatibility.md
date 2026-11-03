# Smart Fan Hardware Compatibility Breakdown

## Executive Overview
AACI (Atomberg Adaptive Comfort Intelligence) aims to run intelligence locally on smart fan hardware (or via connected local edge gateways) while keeping bill-of-materials (BOM) cost and thermal dissipation within commercial ceiling fan specifications.

This audit categorizes all evaluated features across Archive (2), ROBOD, and RDD according to real-world edge feasibility on Atomberg hardware.

---

## Hardware Sensor Classification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AACI HARDWARE SENSING TIERS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Direct On-Board Fan Sensors (Standard / Low BOM)                    │
│   • Ambient Temperature (°C) [e.g., AHT20 / SHT40 / NTC thermistor]          │
│   • Ambient Relative Humidity (%) [e.g., AHT20 / SHT40]                     │
│   • BLDC Motor Speed & Power Telemetry [RPM Hall Sensor, Inverter Shunt]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Additional Sensor Add-on Options (Mid-Tier BOM)                     │
│   • Ambient Illuminance (Lux) [e.g., OPT3001 / BH1750 / Photodiode]         │
│   • Passive Infrared (PIR) Motion Sensor [e.g., AS612 / EKMC]               │
│   • Acoustic Noise Level (dBA) [e.g., I2S MEMS mic / envelope detector]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Specialized High-Value Gas Sensors (Premium BOM)                    │
│   • Carbon Dioxide (CO2 ppm) [e.g., Sensirion SCD40 Photoacoustic NDIR]      │
│   • Total Volatile Organic Compounds (TVOC ppb) [e.g., SGP40]               │
│   • Particulate Matter (PM2.5) [e.g., Sensirion SPS30 Optical]              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Edge / Cloud / Local Network Ingestion                              │
│   • Wi-Fi Associated Device Count (Local Router AP Polling via mDNS/UPnP)   │
│   • Calendar / Schedule Priors (Atomberg Companion App Cloud Sync)          │
│   • Outdoor Weather (OpenWeatherMap API via Wi-Fi)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Feature Feasibility Table

| Feature Name | Source Dataset | Physical Unit | Hardware Category | Typical Component & Interface | BOM Impact | Power / Thermal Considerations | Operational Reliability & Failure Modes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Temperature** | Archive (2), ROBOD, RDD | °C | **Direct Smart-Fan Sensor** | Sensirion SHT40 or Aosong AHT20 (I2C) | Low (<$0.50) | Negligible (<0.5 mW). Mount on bottom canopy away from motor heat dissipation. | High reliability. Motor thermal rise must be compensated via calibration offset. |
| **Relative Humidity** | Archive (2), ROBOD, RDD | % | **Direct Smart-Fan Sensor** | Integrated in SHT40/AHT20 combo sensor (I2C) | Included with Temp sensor | Negligible. | Fast response, slight drift over years in high-grease/dust environments. |
| **Motor Speed / Tach** | RDD, AACI V3 | RPM / Level (1–5) | **Direct Smart-Fan Telemetry** | BLDC Hall effect sensors & back-EMF observer | Zero additional (Native to BLDC driver) | Zero additional. | Fails only on motor stall or inverter bridge fault (trapped by V3.1 reliability layer). |
| **Light (Illuminance)** | Archive (2) | Lux | **Additional Sensor Option** | Texas Instruments OPT3001 or Rohm BH1750 (I2C) | Low (~$0.40) | Negligible (<0.1 mW). Requires translucent window on downrod/under-canopy. | Subject to ambient shadow, curtains, artificial bulbs. Fails during dark occupancy. |
| **CO2 Concentration** | Archive (2), ROBOD | ppm | **Additional / Premium Sensor** | Sensirion SCD40 Photoacoustic NDIR (I2C) | Medium-High ($5.00–$8.00) | Moderate (~15–75 mW pulsed). Requires adequate ventilation slots. | Excellent human metabolic proxy. Slow response time (5–15 min diffusion delay). |
| **Acoustic Level** | ROBOD | dBA | **Additional Sensor Option** | InvenSense ICS-43434 I2S MEMS or analog peak detector | Low-Medium (~$0.80) | Low (<3 mW). Requires acoustic isolation from fan blade aerodynamic whoosh. | Susceptible to TV, music, external traffic false positives. Privacy preserving (RMS only). |
| **Wi-Fi Connected Devices**| ROBOD | Count | **Local Edge / Cloud Proxy** | ESP32 Wi-Fi station / AP probe sniffer or router API | Software only (zero BOM if ESP32 fan) | Zero additional electrical cost. | Privacy concerns if logging MACs; counting anonymous probes is compliant. |
| **Total VOC / PM2.5** | ROBOD | ppb / ug/m3 | **Premium Sensor Option** | SGP40 MOX gas sensor / SPS30 laser scattering | High ($6.00–$15.00) | High (heaters/lasers draw 50–150 mW). | Sensitive to cooking, cleaning agents, aerosols. Secondary proxy only. |
| **HumidityRatio** | Archive (2) | kg/kg dry air | **Derived Feature** | Mathematical calculation: $f(T, RH, P_{atm})$ | Zero (Firmware math) | Minimal MCU compute (10 floating-point ops). | 100% reliable if T and RH are valid. |
| **Time-of-day / Day-of-week**| All | Cyclic sin/cos | **Derived Feature** | On-chip RTC or NTP sync over Wi-Fi | Zero | Zero. | Extremely reliable. Requires periodic NTP sync if power cycled. |

---

## Strategic Recommendation for AACI Model 2

### Feature Set Strategy:
1. **Set A (All Available Features - Theoretical Upper Bound)**:
   - Includes: `{Temperature, Humidity, Light, CO2, HumidityRatio, Hour_sin, Hour_cos}`
   - Serves as the high-accuracy benchmark for premium setups or IoT sensor hubs.
2. **Set B (Minimal Low-Cost Smart Fan BOM - Hardware Deployable)**:
   - Includes: `{Temperature, Humidity, HumidityRatio, Hour_sin, Hour_cos, (Optional Light)}`
   - Evaluates whether room occupancy can still be detected accurately *without* costly CO2 sensors, relying on micro-thermal shifts, humidity accumulation from occupant respiration/perspiration, and diurnal occupancy priors.

This two-tier evaluation guarantees that Atomberg engineering gets both the scientific performance ceiling and the realistic production-line baseline.

