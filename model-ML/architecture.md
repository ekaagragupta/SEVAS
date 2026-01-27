```

┌─────────────────────────────────────────────────┐
│          USER UPLOADS SATELLITE IMAGE            │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│          PREPROCESSING (OpenCV)                  │
│  • Resize to 256x256                            │
│  • Remove clouds                                │
│  • Normalize colors                             │
│  • Extract NDVI (vegetation index)              │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ↓                 ↓
┌──────────────┐  ┌──────────────────┐
│ QUICK PATH   │  │  DETAILED PATH   │
│ (2-3 sec)    │  │  (30-60 sec)     │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       ↓                   ↓
┌──────────────┐  ┌──────────────────┐
│  Vision AI   │  │  U-Net Model     │
│  (Gemini)    │  │  (Custom CNN)    │
│              │  │                  │
│ Output:      │  │ Output:          │
│ "Sand mining │  │ Pixel mask:      │
│  detected in │  │ 🔴🔴🔴 = mining   │
│  NE section" │  │ 🟢🟢🟢 = healthy  │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       └────────┬──────────┘
                │
                ↓
┌─────────────────────────────────────────────────┐
│          COMBINE RESULTS                         │
│  • Vision AI gives context                      │
│  • U-Net gives precise locations                │
│  • Calculate risk percentage                    │
│  • Extract GPS coordinates                      │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│      TEMPORAL ANALYSIS (if history available)   │
│      LSTM checks: Is this getting worse?        │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│          FINAL OUTPUT                            │
│  • Highlighted image                            │
│  • Risk percentage: 87%                         │
│  • GPS: 26.9124°N, 75.7873°E                   │
│  • Prediction: "Likely to worsen"              │
│  • Report: Auto-generated PDF                   │
└─────────────────────────────────────────────────┘
