## ** System Architecture (How It All Connects)**

```
┌─────────────────────────────────────────────────────────────┐
│                    LANDSENTINEL PLATFORM                     │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Data Sources │
└──────┬───────┘
       │
       ├─ Sentinel-2 Satellite (Multi-region, Multi-temporal)
       ├─ Landsat 8/9 (Historical data)
       ├─ Drone Imagery (Ground truth)
       └─ Citizen Reports (Mobile uploads)
       │
       ↓
┌──────────────────────────────────────────┐
│      AI/ML PROCESSING PIPELINE           │
├──────────────────────────────────────────┤
│                                          │
│  Module 1: PREDICTIVE ENGINE             │
│  ├─ LSTM Temporal Analysis               │
│  ├─ Multi-spectral Index Tracking        │
│  └─ Anomaly Detection                    │
│      Output: Risk predictions            │
│                                          │
│  Module 2: DETECTION ENGINE              │
│  ├─ U-Net Segmentation Model             │
│  ├─ Vision AI (Gemini/GPT-4V)            │
│  └─ Change Detection (Siamese CNN)       │
│      Output: Violation locations         │
│                                          │
│  Module 3: CROSS-BORDER INTELLIGENCE     │
│  ├─ Equipment Signature Matching         │
│  ├─ Temporal Correlation Analysis        │
│  └─ Network Graph Construction           │
│      Output: Criminal network maps       │
│                                          │
│  Module 4: PRIORITIZATION ENGINE         │
│  ├─ Multi-factor Severity Scoring        │
│  ├─ Route Optimization (Dijkstra)        │
│  └─ Resource Allocation                  │
│      Output: Ranked action plan          │
│                                          │
└──────────────┬───────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────┐
│       BACKEND API (Node.js)              │
├──────────────────────────────────────────┤
│  - Case management                       │
│  - Multi-region data aggregation         │
│  - Jurisdictional routing                │
│  - Forensic report generation            │
│  - Alert distribution                    │
└──────────────┬───────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────┐
│      DATABASE (MySQL)                    │
├──────────────────────────────────────────┤
│  Tables:                                 │
│  - violations (all detected cases)       │
│  - predictions (risk forecasts)          │
│  - network_clusters (cross-border links) │
│  - inspection_logs (field team actions)  │
│  - jurisdictions (authority mapping)     │
└──────────────┬───────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────┐
│   FRONTEND DASHBOARD (Next.js)           │
├──────────────────────────────────────────┤
│                                          │
│  View 1: PREDICTIVE MAP                  │
│  - Heat map of predicted high-risk zones │
│  - Timeline: "Next 2 weeks forecast"     │
│                                          │
│  View 2: PRIORITY QUEUE                  │
│  - Ranked list: Critical → Low           │
│  - Optimized inspection routes           │
│                                          │
│  View 3: NETWORK INTELLIGENCE            │
│  - Multi-region overview                 │
│  - Correlation graphs                    │
│  - Cluster visualization                 │
│                                          │
│  View 4: FORENSIC REPORTS                │
│  - Timeline reconstruction               │
│  - Damage quantification                 │
│  - GPS-tagged evidence                   │
│                                          │
└──────────────────────────────────────────┘
```

