# SEVAS API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:5000`  
**Protocol:** HTTP/REST  
**Content-Type:** `application/json`

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)
5. [Endpoints](#endpoints)
   - [Health & Info](#health--info)
   - [Analysis](#analysis)
   - [Preprocessing](#preprocessing)
   - [Segmentation](#segmentation)
6. [Response Formats](#response-formats)
7. [Code Examples](#code-examples)

---

## Overview

The SEVAS API provides RESTful endpoints for environmental violation detection using satellite imagery. The API supports:

- ✅ Single image analysis
- ✅ Batch processing
- ✅ Multiple detection types (sand mining, land encroachment, vegetation loss)
- ✅ Spectral index calculation (NDVI, NDWI)
- ✅ U-Net semantic segmentation
- ✅ Cloud detection and filtering

### Supported File Formats

| Format | Extension | Max Size | Notes |
|--------|-----------|----------|-------|
| JPEG | `.jpg`, `.jpeg` | 50MB | Recommended for general use |
| PNG | `.png` | 50MB | Lossless, larger files |
| TIFF | `.tif`, `.tiff` | 50MB | Geospatial data support |
| GeoTIFF | `.geotiff` | 50MB | Georeferenced imagery |

---

## Authentication

**Current Version:** No authentication required (development mode)

**Production:** API key authentication will be required:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://api.sevas.com/api/analyze
```

---

## Error Handling

### Standard Error Response
```json
{
  "status": "error",
  "message": "Detailed error description",
  "code": "ERROR_CODE"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input or missing parameters |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File exceeds 50MB limit |
| 500 | Internal Server Error | Server-side error |

### Common Error Codes
```json
{
  "NO_FILE_PROVIDED": "No file uploaded",
  "INVALID_FILE_TYPE": "Unsupported file format",
  "FILE_TOO_LARGE": "File exceeds maximum size",
  "MODEL_NOT_FOUND": "Trained model not available",
  "PROCESSING_ERROR": "Error during image processing"
}
```

---

## Rate Limiting

**Development:** No rate limits

**Production:**
- 100 requests per minute per IP
- 1000 requests per day per API key

---

## Endpoints

### Health & Info

#### GET `/api/health`

Health check endpoint to verify API status.

**Request:**
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "SEVAS ML API",
  "version": "1.0.0",
  "timestamp": "2026-03-06T16:30:00.000Z"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `500 Internal Server Error` - Service is down

---

#### GET `/api/info`

Get comprehensive service information.

**Request:**
```bash
curl http://localhost:5000/api/info
```

**Response:**
```json
{
  "project": "SEVAS",
  "full_name": "Satellite-based Environmental Violation Analysis System",
  "description": "AI-powered detection of sand mining and land encroachment",
  "version": "1.0.0",
  "capabilities": [
    "Sand mining detection",
    "Land encroachment detection",
    "Vegetation loss monitoring",
    "Cloud detection",
    "Change detection",
    "Spectral indices (NDVI, NDWI)"
  ],
  "endpoints": {
    "analysis": "/api/analyze",
    "batch": "/api/batch",
    "segment": "/api/segment",
    "preprocess": "/api/preprocess",
    "spectral": "/api/spectral-indices"
  }
}
```

---

#### GET `/api/models`

List available detection models and capabilities.

**Request:**
```bash
curl http://localhost:5000/api/models
```

**Response:**
```json
{
  "detection_types": [
    {
      "id": "general",
      "name": "General Environmental Analysis",
      "description": "Comprehensive analysis for all violation types"
    },
    {
      "id": "sand_mining",
      "name": "Sand Mining Detection",
      "description": "Specialized detection of illegal sand extraction"
    },
    {
      "id": "land_encroachment",
      "name": "Land Encroachment Detection",
      "description": "Detection of unauthorized construction"
    },
    {
      "id": "vegetation",
      "name": "Vegetation Loss Detection",
      "description": "Monitoring of deforestation and land clearing"
    }
  ],
  "preprocessing": [
    "NDVI (Normalized Difference Vegetation Index)",
    "NDWI (Normalized Difference Water Index)",
    "Cloud Detection",
    "Change Detection"
  ]
}
```

---

### Analysis

#### POST `/api/analyze`

Analyze a single satellite image for environmental violations.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file to analyze |
| `detection_type` | String | No | Type of analysis (default: `general`) |
| `include_preprocessing` | Boolean | No | Include preprocessing results (default: `false`) |

**Detection Types:**
- `general` - All violation types
- `sand_mining` - Sand mining specific
- `land_encroachment` - Land encroachment specific
- `vegetation` - Vegetation loss specific

**Request Example:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@satellite_image.jpg" \
  -F "detection_type=sand_mining" \
  -F "include_preprocessing=true"
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "analysis_id": "abc12345",
    "timestamp": "2026-03-06T16:30:00.000Z",
    "filename": "satellite_image.jpg",
    "detection_type": "sand_mining",
    "analysis": {
      "violations_detected": true,
      "confidence": "High",
      "severity": "Moderate",
      "location": "Northeastern riverbank section, approximately 500m from bridge",
      "summary": "Sand mining activity detected with disturbed riverbed patterns and visible excavation marks. Estimated affected area: 2.3 hectares.",
      "recommendations": [
        "Immediate field verification recommended",
        "Coordinate with local authorities",
        "Monitor for expansion over next 2 weeks"
      ],
      "full_response": "Detailed AI analysis text..."
    },
    "preprocessing": {
      "ndvi": {
        "mean": 0.234,
        "min": -0.156,
        "max": 0.789
      },
      "ndwi": {
        "mean": 0.123,
        "min": -0.234,
        "max": 0.567
      },
      "cloud_coverage": 12.5,
      "image_usable": true
    }
  }
}
```

**Error Response (400 Bad Request):**
```json
{
  "status": "error",
  "message": "No file provided"
}
```

---

#### POST `/api/batch`

Process multiple images in a single request.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `files` | File[] | Yes | Multiple image files |
| `detection_type` | String | No | Type of analysis (default: `general`) |

**Request Example:**
```bash
curl -X POST http://localhost:5000/api/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "detection_type=general"
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "batch_id": "batch_xyz789",
  "total_files": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "analysis_id": "batch_xyz789_1",
      "violations_detected": true,
      "confidence": "High",
      "severity": "Severe",
      "summary": "Sand mining detected..."
    },
    {
      "filename": "image2.jpg",
      "analysis_id": "batch_xyz789_2",
      "violations_detected": false,
      "confidence": "High",
      "severity": "None",
      "summary": "No violations detected..."
    },
    {
      "filename": "image3.jpg",
      "error": "Invalid file format"
    }
  ]
}
```

---

#### GET `/api/results/{analysis_id}`

Retrieve results from a previous analysis.

**Request Example:**
```bash
curl http://localhost:5000/api/results/abc12345
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "analysis_id": "abc12345",
    "timestamp": "2026-03-06T16:30:00.000Z",
    "analysis": { ... }
  }
}
```

**Error Response (404 Not Found):**
```json
{
  "status": "error",
  "message": "Analysis ID not found"
}
```

---

### Preprocessing

#### POST `/api/preprocess`

Preprocess an image without running detection.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file to preprocess |

**Request Example:**
```bash
curl -X POST http://localhost:5000/api/preprocess \
  -F "file=@image.jpg"
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "original_shape": [3252, 4878, 3],
  "processed_shape": [256, 256, 3],
  "value_range": {
    "min": 0.0,
    "max": 1.0
  }
}
```

---

#### POST `/api/spectral-indices`

Calculate spectral indices (NDVI, NDWI) for an image.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file to analyze |

**Request Example:**
```bash
curl -X POST http://localhost:5000/api/spectral-indices \
  -F "file=@satellite.jpg"
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "ndvi": {
    "mean": 0.234,
    "min": -0.156,
    "max": 0.789,
    "median": 0.245,
    "interpretation": "Moderate vegetation"
  },
  "ndwi": {
    "mean": 0.123,
    "min": -0.234,
    "max": 0.567,
    "median": 0.134,
    "interpretation": "Some water bodies present"
  }
}
```

**NDVI Interpretation:**

| Range | Interpretation |
|-------|----------------|
| 0.6 to 1.0 | Dense vegetation |
| 0.2 to 0.6 | Moderate vegetation |
| -0.1 to 0.2 | Bare soil / sparse vegetation |
| -1.0 to -0.1 | Water / clouds / snow |

**NDWI Interpretation:**

| Range | Interpretation |
|-------|----------------|
| 0.3 to 1.0 | Water bodies |
| -0.3 to 0.3 | Vegetation / land |
| -1.0 to -0.3 | Dry / bare land |

---

### Segmentation

#### POST `/api/segment`

Perform U-Net semantic segmentation on an image.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file to segment |

**Request Example:**
```bash
curl -X POST http://localhost:5000/api/segment \
  -F "file=@satellite.jpg"
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "segmentation_image": "segmentation_20260306_163000.png",
  "timestamp": "2026-03-06T16:30:00.000Z",
  "analysis": {
    "total_pixels": 65536,
    "class_distribution": {
      "Background": {
        "pixels": 12500,
        "percentage": 19.07
      },
      "Sand Mining": {
        "pixels": 15000,
        "percentage": 22.89
      },
      "Vegetation": {
        "pixels": 30000,
        "percentage": 45.78
      },
      "Water": {
        "pixels": 8036,
        "percentage": 12.26
      }
    },
    "violation_detected": true,
    "severity": "High"
  }
}
```

**Segmentation Classes:**

| Class ID | Class Name | Color | Description |
|----------|------------|-------|-------------|
| 0 | Background | Gray | Normal land / unclassified |
| 1 | Sand Mining | Red | Mining / disturbance areas |
| 2 | Vegetation | Green | Healthy vegetation |
| 3 | Water | Blue | Water bodies |

**Error Response (404 Not Found):**
```json
{
  "status": "error",
  "message": "U-Net model not found. Train the model first."
}
```

---

## Response Formats

### Analysis Response Object
```typescript
interface AnalysisResponse {
  status: "success" | "error";
  data?: {
    analysis_id: string;
    timestamp: string;
    filename: string;
    detection_type: string;
    analysis: {
      violations_detected: boolean | null;
      confidence: "High" | "Medium" | "Low";
      severity: "None" | "Minor" | "Moderate" | "Severe";
      location: string;
      summary: string;
      recommendations: string[];
      full_response: string;
    };
    preprocessing?: {
      ndvi: SpectralIndex;
      ndwi: SpectralIndex;
      cloud_coverage: number;
      image_usable: boolean;
    };
  };
  message?: string;  // Only present on error
}

interface SpectralIndex {
  mean: number;
  min: number;
  max: number;
  median?: number;
}
```

---

## Code Examples

### Python
```python
import requests

# Single image analysis
def analyze_image(image_path, detection_type='general'):
    url = 'http://localhost:5000/api/analyze'
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'detection_type': detection_type,
            'include_preprocessing': 'true'
        }
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print(f"Violations: {result['data']['analysis']['violations_detected']}")
        print(f"Severity: {result['data']['analysis']['severity']}")
        print(f"Summary: {result['data']['analysis']['summary']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Usage
result = analyze_image('satellite_image.jpg', 'sand_mining')
```

### JavaScript (Node.js)
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function analyzeImage(imagePath, detectionType = 'general') {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));
  form.append('detection_type', detectionType);
  
  try {
    const response = await axios.post(
      'http://localhost:5000/api/analyze',
      form,
      { headers: form.getHeaders() }
    );
    
    const { data } = response.data;
    console.log('Violations:', data.analysis.violations_detected);
    console.log('Severity:', data.analysis.severity);
    console.log('Summary:', data.analysis.summary);
    
    return data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    return null;
  }
}

// Usage
analyzeImage('satellite_image.jpg', 'sand_mining');
```

### cURL
```bash
# Basic analysis
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@satellite_image.jpg" \
  -F "detection_type=sand_mining"

# With preprocessing
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@satellite_image.jpg" \
  -F "detection_type=general" \
  -F "include_preprocessing=true"

# Batch processing
curl -X POST http://localhost:5000/api/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"

# U-Net segmentation
curl -X POST http://localhost:5000/api/segment \
  -F "file=@satellite.jpg"

# Spectral indices
curl -X POST http://localhost:5000/api/spectral-indices \
  -F "file=@satellite.jpg"
```

---

## Best Practices

### Image Quality

✅ **Recommended:**
- Resolution: 256x256 or higher
- Format: JPEG or PNG
- Color space: RGB
- File size: < 10MB for faster processing

❌ **Avoid:**
- Very low resolution (< 128x128)
- Grayscale images (RGB preferred)
- Heavily compressed images
- Corrupted or incomplete files

### API Usage

✅ **Do:**
- Check `/api/health` before batch operations
- Use appropriate `detection_type` for specific violations
- Handle errors gracefully
- Cache results using `analysis_id`

❌ **Don't:**
- Send the same file repeatedly
- Upload files > 50MB
- Make concurrent requests without rate limiting

---

## Changelog

### Version 1.0.0 (2026-03-06)
- ✅ Initial API release
- ✅ Basic analysis endpoints
- ✅ U-Net segmentation
- ✅ Spectral indices calculation
- ✅ Batch processing support

---


**API Version:** 1.0.0