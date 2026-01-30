# Advanced Preprocessing

- NDVI Calculator (Vegetation Health)
- NDWI Calculator (Water Body Detection)
- Cloud Detection & Removal
- Change Detection Preprocessing

## 📊 NDVI - Normalized Difference Vegetation Index

## Satellite (Sentinel-2) Data

🛰️ → Red, Green, Blue, Near-Infrared, Short-wave Infrared, etc. (13 channels!)

### Why Multiple Bands?

Different materials reflect light differently in different wavelengths:

**🌿 Healthy Vegetation:**

- Absorbs RED light (for photosynthesis)
- Reflects NEAR-INFRARED strongly
- Result: High NIR, Low Red

**🏜️ Bare Soil/Sand:**

- Reflects both Red and NIR equally
- Result: Similar NIR and Red

**💧 Water:**

- Absorbs NIR strongly
- Reflects visible light
- Result: Low NIR, Higher visible

## Formula

```
NDVI = (NIR - Red) / (NIR + Red)
```

### Values range from -1 to +1:

- +0.6 to +1.0 = Dense, healthy vegetation 🌳🌲
- +0.2 to +0.6 = Moderate vegetation 🌱
- -0.1 to +0.2 = Bare soil, rocks 🏜️
- -1.0 to -0.1 = Water, snow, clouds 💧❄️

## Why We Need NDVI for SEVAS

### Detecting Vegetation Loss

Before mining: NDVI = 0.7 (healthy forest)
After mining: NDVI = 0.1 (bare land)
→ 85% vegetation loss detected! ⚠️

### Detecting Land Clearing

- Month 1: NDVI = 0.6
- Month 2: NDVI = 0.5
- Month 3: NDVI = 0.2
  → Progressive clearing detected! 🚨

# 💧 NDWI - Normalized Difference Water Index

## What Is NDWI?

### Formula

```
NDWI = (Green - NIR) / (Green + NIR)
```

### Values

- +0.3 to +1.0 = Water bodies 💧
- -0.3 to +0.3 = Vegetation/Land 🌍
- -1.0 to -0.3 = Dry land 🏜️

## Why We Need NDWI for SEVAS

### Detecting Sand Mining in Rivers

Before: NDWI = 0.5 (water)
After: NDWI = -0.2 (exposed sand)
→ Riverbed excavation detected! ⚠️

### Tracking Riverbank Changes

- Year 1: Water area = 100 hectares
- Year 2: Water area = 85 hectares
  → 15% water loss = possible mining 🚨
