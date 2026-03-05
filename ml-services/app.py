
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Import our modules
from utils.image_processor import ImageProcessor
from utils.spectral_indices import SpectralIndices
from utils.cloud_detector import CloudDetector
from utils.change_detector import ChangeDetector

# Try to import Vision AI, fall back to mock if not available
try:
    from models.vision_ai import VisionAI
    vision_ai = VisionAI()
    if not hasattr(vision_ai, 'gemini_model') or vision_ai.gemini_model is None:
        from models.vision_ai import VisionAI
        vision_ai = VisionAI()
except:
    from models.vision_ai import VisionAI
    vision_ai = VisionAI()

# Load environment
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'geotiff'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize modules
processor = ImageProcessor(target_size=256)
spectral = SpectralIndices()
cloud_detector = CloudDetector()
change_detector = ChangeDetector()

# Store for analysis results (in production, use a database)
analysis_results = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_id():
    """Generate unique ID for analysis"""
    return str(uuid.uuid4())[:8]




@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "SEVAS ML API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/info', methods=['GET'])
def service_info():
    """Service information"""
    return jsonify({
        "project": "SEVAS",
        "full_name": "Satellite-based Environmental Violation Analysis System",
        "description": "AI-powered detection of sand mining and land encroachment",
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
            "preprocess": "/api/preprocess",
            "spectral": "/api/spectral-indices",
            "health": "/api/health"
        }
    }), 200


@app.route('/api/models', methods=['GET'])
def available_models():
    """List available detection models"""
    return jsonify({
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
    }), 200




@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Main analysis endpoint
    
    Request:
        - file: Image file (multipart/form-data)
        - detection_type: Type of analysis (optional, default: general)
        - include_preprocessing: Include preprocessing results (optional)
    
    Response:
        - analysis_id: Unique ID for this analysis
        - results: Analysis results
        - preprocessing: Preprocessing results (if requested)
    """
    print("\n" + "="*70)
    print("🔍 NEW ANALYSIS REQUEST")
    print("="*70)
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file provided"
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": f"Invalid file type. Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
        }), 400
    
    try:
        # Generate unique ID
        analysis_id = generate_id()
        
        # Get parameters
        detection_type = request.form.get('detection_type', 'general')
        include_preprocessing = request.form.get('include_preprocessing', 'false').lower() == 'true'
        
        print(f"Analysis ID: {analysis_id}")
        print(f"Detection type: {detection_type}")
        print(f"Include preprocessing: {include_preprocessing}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{analysis_id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        print(f"✅ File saved: {filepath}")
        
        # Initialize result structure
        result = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "detection_type": detection_type
        }
        
        # Run preprocessing if requested
        preprocessing_results = {}
        if include_preprocessing:
            print("\n📊 Running preprocessing...")
            
            # Load and preprocess image
            img = processor.load_image(filepath)
            if img is not None:
                img_resized = processor.resize_image(img)
                
                # Calculate spectral indices
                ndvi = spectral.calculate_ndvi(img_resized)
                ndwi = spectral.calculate_ndwi(img_resized)
                
                # Cloud detection
                cloud_mask, cloud_pct = cloud_detector.detect_clouds(img_resized)
                is_usable = cloud_detector.is_image_usable(cloud_pct)
                
                preprocessing_results = {
                    "ndvi": {
                        "mean": float(ndvi.mean()) if ndvi is not None else None,
                        "min": float(ndvi.min()) if ndvi is not None else None,
                        "max": float(ndvi.max()) if ndvi is not None else None
                    },
                    "ndwi": {
                        "mean": float(ndwi.mean()) if ndwi is not None else None,
                        "min": float(ndwi.min()) if ndwi is not None else None,
                        "max": float(ndwi.max()) if ndwi is not None else None
                    },
                    "cloud_coverage": float(cloud_pct),
                    "image_usable": bool(is_usable)
                }
                
                result["preprocessing"] = preprocessing_results
        
        # Run AI analysis
        print("\n🤖 Running AI analysis...")
        analysis = vision_ai.analyze_with_gemini(filepath, detection_type)
        
        # Format results
        result["analysis"] = {
            "violations_detected": analysis.get("violations_detected"),
            "confidence": analysis.get("confidence"),
            "severity": analysis.get("severity"),
            "location": analysis.get("location"),
            "summary": analysis.get("summary"),
            "recommendations": analysis.get("recommendations"),
            "full_response": analysis.get("raw_response")
        }
        
        # Store result
        analysis_results[analysis_id] = result
        
        print("\n Analysis complete!")
        print("="*70)
        
        return jsonify({
            "status": "success",
            "data": result
        }), 200
        
    except Exception as e:
        print(f"\n  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# BATCH ANALYSIS ENDPOINT


@app.route('/api/batch', methods=['POST'])
def batch_analyze():
   
    print(" BATCH ANALYSIS REQUEST")

    
    if 'files' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No files provided"
        }), 400
    
    files = request.files.getlist('files')
    detection_type = request.form.get('detection_type', 'general')
    
    print(f"Number of files: {len(files)}")
    print(f"Detection type: {detection_type}")
    
    batch_id = generate_id()
    results = []
    
    for idx, file in enumerate(files, 1):
        print(f"\n--- Processing file {idx}/{len(files)} ---")
        
        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                saved_filename = f"{batch_id}_{idx}_{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(filepath)
                
                # Analyze
                analysis = vision_ai.analyze_with_gemini(filepath, detection_type)
                
                results.append({
                    "filename": filename,
                    "analysis_id": f"{batch_id}_{idx}",
                    "violations_detected": analysis.get("violations_detected"),
                    "confidence": analysis.get("confidence"),
                    "severity": analysis.get("severity"),
                    "summary": analysis.get("summary")
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        else:
            results.append({
                "filename": file.filename,
                "error": "Invalid file type"
            })
    
    print(f"\n Batch processing complete: {len(results)} files")
    print("="*70)
    
    return jsonify({
        "status": "success",
        "batch_id": batch_id,
        "total_files": len(files),
        "results": results
    }), 200

# PREPROCESSING ENDPOINTS


@app.route('/api/preprocess', methods=['POST'])
def preprocess_image():
  
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(filepath)
        
        # Preprocess
        img = processor.load_image(filepath)
        img_resized = processor.resize_image(img)
        img_normalized = processor.normalize_image(img_resized)
        
        result = {
            "status": "success",
            "original_shape": img.shape if img is not None else None,
            "processed_shape": img_normalized.shape if img_normalized is not None else None,
            "value_range": {
                "min": float(img_normalized.min()) if img_normalized is not None else None,
                "max": float(img_normalized.max()) if img_normalized is not None else None
            }
        }
        
        # Clean up temp file
        os.remove(filepath)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/spectral-indices', methods=['POST'])
def calculate_spectral_indices():
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(filepath)
        
        # Process
        img = processor.load_image(filepath)
        img_resized = processor.resize_image(img)
        
        # Calculate indices
        ndvi = spectral.calculate_ndvi(img_resized)
        ndwi = spectral.calculate_ndwi(img_resized)
        
        result = {
            "status": "success",
            "ndvi": {
                "mean": float(ndvi.mean()),
                "min": float(ndvi.min()),
                "max": float(ndvi.max()),
                "median": float(np.median(ndvi))
            },
            "ndwi": {
                "mean": float(ndwi.mean()),
                "min": float(ndwi.min()),
                "max": float(ndwi.max()),
                "median": float(np.median(ndwi))
            }
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/api/results/<analysis_id>', methods=['GET'])
def get_results(analysis_id):
    """Get results by analysis ID"""
    if analysis_id in analysis_results:
        return jsonify({
            "status": "success",
            "data": analysis_results[analysis_id]
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Analysis ID not found"
        }), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "status": "error",
        "message": "File too large. Maximum size: 50MB"
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500



if __name__ == '__main__':
   
    print(f"Service: SEVAS - Environmental Violation Detection")
    print(f"Version: 1.0.0")
    print(f"Port: 5000")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("="*70)
    print("\n📋 Available endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/info")
    print("  GET  /api/models")
    print("  POST /api/analyze")
    print("  POST /api/batch")
    print("  POST /api/preprocess")
    print("  POST /api/spectral-indices")
    print("  GET  /api/results/<id>")
    print("\n Server running at: http://localhost:5000")
   
    
    app.run(debug=True, port=5000, host='0.0.0.0')

# U-NET SEGMENTATION ENDPOINT


@app.route('/api/segment', methods=['POST'])
def segment_image():
    """
    U-Net segmentation endpoint
    
    Request:
        - file: Image file
    
    Response:
        - segmentation_mask: URL to segmentation result
        - analysis: Pixel-wise analysis
        - violations: Detected violations
    """
    print("\n" + "="*70)
    print(" U-NET SEGMENTATION REQUEST")
    print("="*70)
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file"}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"segment_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Try to load U-Net model and predict
        try:
            from models.predict_unet import UNetPredictor
            
            predictor = UNetPredictor('models/saved_models/unet_best.h5')
            image, mask, confidence = predictor.predict_image(filepath)
            
            # Save visualization
            output_filename = f"segmentation_{timestamp}.png"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            predictor.visualize_prediction(image, mask, confidence, save_path=output_path)
            
            # Analyze
            analysis = predictor.analyze_prediction(mask)
            
            result = {
                "status": "success",
                "segmentation_image": output_filename,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            print(" Segmentation complete!")
            
            return jsonify(result), 200
            
        except FileNotFoundError:
            return jsonify({
                "status": "error",
                "message": "U-Net model not found. Train the model first."
            }), 404
            
    except Exception as e:
        print(f" Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500