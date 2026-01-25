# Problem Statement

Environmental crimes like illegal sand mining and land encroachment operate at scale across jurisdictional boundaries, with violators strategically rotating operations.

Government enforcement agencies face three critical challenges:
1. Detecting violations too late to prevent damage.
2. Being overwhelmed by hundreds of low-priority alerts while critical cases go unaddressed.
3. Lacking visibility into cross-border operations where the same perpetrators exploit jurisdictional gaps.

## LandSentinel addresses this through an AI-powered predictive enforcement platform that:

1. **Predicts violations 2–3 weeks** in advance using LSTM-based temporal pattern analysis across multi-spectral satellite imagery.
2. **Intelligently prioritizes cases** through a multi-factor severity scoring engine.
3. **Identifies cross-jurisdictional criminal networks** by correlating violation patterns across state/national borders.

The system processes satellite data across multiple regions simultaneously, generates optimized inspection routes for field teams, and auto-routes alerts to the appropriate authorities.
<img width="888" height="605" alt=" - visual selection" src="https://github.com/user-attachments/assets/77f4a73b-d6c8-43a2-a388-e8df18e63cc0" />

# 📊 PROBLEM BREAKDOWN

### Component A : Predictive early warning system 
Tehnology stack:
- **LSTM Network** for temporal sequence analysis
- **Multi spectral indices** (NDVI , NDWI, NDBI) tracking
- **Anomaly detection** algorithm for suspicious patterns
- **Time series forecasting** for risk prediction

Mechanisms :

  
Input: 6 months of satellite imagery (monthly)
      ↓
Extract temporal features:
  - Vegetation health trends (NDVI decline)
  - Water body area changes (NDWI)
  - Built-up area expansion (NDBI)
  - Vehicle detection counts
  - Night-time light intensity
      ↓
LSTM Model analyzes patterns
      ↓
Output: Risk prediction 2-3 weeks ahead
  - Probability: 78% chance of violation
  - Warning signs: "Vegetation decline 23%, vehicle activity +340%"
  - Predicted timeline: "High risk period: Jan 15-30"

   
