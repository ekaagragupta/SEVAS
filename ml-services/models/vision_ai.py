"""
Vision AI Module for SEVAS
Integrates Gemini and OpenAI Vision APIs for image analysis
"""

import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class VisionAI:
    """
    Wrapper for Vision AI APIs (Gemini and OpenAI)
    Analyzes satellite images for environmental violations
    """
    
    def __init__(self):
        """Initialize Vision AI with API keys"""
        
        # Get API keys from environment
        self.gemini_key = os.getenv('GEMINI_API_KEY')
       
        # Configure Gemini
        if self.gemini_key and self.gemini_key != 'your_actual_gemini_key_here':
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini Vision AI initialized")
        else:
            self.gemini_model = None
            print(" Gemini API key not configured")
        
       
    
    def analyze_with_gemini(self, image_path, detection_type="general"):
        """
        Analyze image using Gemini Vision
        
        Args:
            image_path (str): Path to image file
            detection_type (str): Type of analysis
                - "general": General environmental analysis
                - "sand_mining": Focus on sand mining detection
                - "land_encroachment": Focus on illegal construction
                - "vegetation": Focus on vegetation loss
                
        Returns:
            dict: Analysis results with description, violations, confidence
        """
        print(f"\n🤖 Analyzing image with Gemini Vision AI...")
        print(f"   Detection type: {detection_type}")
        
        if not self.gemini_model:
            return {
                "error": "Gemini API not configured",
                "description": None
            }
        
        try:
            # Load image
            from PIL import Image
            img = Image.open(image_path)
            
            # Create specialized prompt based on detection type
            prompt = self._create_prompt(detection_type)
            
            print(f"   Sending request to Gemini...")
            
            # Generate response
            response = self.gemini_model.generate_content([prompt, img])
            
            # Parse response
            analysis = self._parse_gemini_response(response.text, detection_type)
            
            print(f"✅ Analysis complete!")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error with Gemini API: {str(e)}")
            return {
                "error": str(e),
                "description": None
            }
    
    def _create_prompt(self, detection_type):
        """
        Create specialized prompt based on detection type
        
        This is PROMPT ENGINEERING - very important!
        The quality of the prompt determines the quality of the response
        """
        
        if detection_type == "sand_mining":
            prompt = """
You are an expert environmental analyst specializing in detecting illegal sand mining from satellite imagery.

Analyze this satellite/aerial image for signs of SAND MINING activity:

WHAT TO LOOK FOR:
1. Disturbed riverbed patterns (unnatural excavation marks)
2. Color changes in river/water bodies (exposed sand/sediment)
3. Vehicle tracks near water bodies
4. Accumulation of extracted material (sand piles)
5. Changes in water flow patterns
6. Vegetation removal along riverbanks

PROVIDE:
1. Is there evidence of sand mining? (Yes/No/Uncertain)
2. Location in image (e.g., "northeastern section", "along the riverbank")
3. Estimated affected area (in hectares if visible, or small/medium/large)
4. Confidence level (High/Medium/Low)
5. Specific indicators you observed
6. Severity assessment (Minor/Moderate/Severe)

Be specific and factual. If no sand mining is detected, clearly state that.
"""
        
        elif detection_type == "land_encroachment":
            prompt = """
You are an expert environmental analyst detecting illegal land encroachment and unauthorized construction.

Analyze this satellite/aerial image for LAND ENCROACHMENT:

WHAT TO LOOK FOR:
1. New unauthorized structures/buildings
2. Construction on forest land or protected areas
3. Road construction in restricted zones
4. Cleared land for development
5. Boundary violations (if reference boundaries visible)

PROVIDE:
1. Evidence of encroachment? (Yes/No/Uncertain)
2. Type of encroachment (construction/clearing/road building)
3. Location and estimated area affected
4. Confidence level (High/Medium/Low)
5. Specific indicators observed
6. Severity assessment

Be precise and objective.
"""
        
        elif detection_type == "vegetation":
            prompt = """
You are an environmental analyst specializing in vegetation loss and deforestation detection.

Analyze this satellite/aerial image for VEGETATION LOSS:

WHAT TO LOOK FOR:
1. Areas of cleared forest or vegetation
2. Comparison of vegetation density across the image
3. Bare soil or exposed land
4. Signs of logging or clearing activity
5. Vegetation health indicators

PROVIDE:
1. Evidence of vegetation loss? (Yes/No/Uncertain)
2. Extent of loss (area affected)
3. Pattern (systematic clearing vs. natural)
4. Confidence level
5. Possible cause (logging, agriculture, fire, etc.)
6. Environmental impact assessment

Be thorough and scientific.
"""
        
        else:  # general
            prompt = """
You are an expert environmental analyst reviewing satellite imagery for environmental violations.

Analyze this image for ANY of the following:
1. SAND MINING: Disturbed riverbeds, excavation, sand extraction
2. LAND ENCROACHMENT: Unauthorized construction, clearing
3. VEGETATION LOSS: Deforestation, land clearing
4. RIVERBANK CHANGES: Erosion, unauthorized modifications

PROVIDE A COMPREHENSIVE ANALYSIS:
1. What violations are visible? (if any)
2. Where in the image? (location description)
3. How severe? (Minor/Moderate/Severe)
4. Estimated area affected
5. Confidence in detection (High/Medium/Low)
6. Key indicators that led to this assessment
7. Recommendations for field verification

If NO violations are detected, clearly state the image shows normal/natural land use.

Be objective, specific, and actionable.
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text, detection_type):
        """
        Parse Gemini's text response into structured data
        
        Args:
            response_text (str): Raw response from Gemini
            detection_type (str): Type of detection performed
            
        Returns:
            dict: Structured analysis results
        """
        
        # Extract key information from response
        analysis = {
            "raw_response": response_text,
            "detection_type": detection_type,
            "violations_detected": self._detect_violations(response_text),
            "summary": self._extract_summary(response_text),
            "confidence": self._extract_confidence(response_text),
            "severity": self._extract_severity(response_text),
            "location": self._extract_location(response_text),
            "recommendations": self._extract_recommendations(response_text)
        }
        
        return analysis
    
    def _detect_violations(self, text):
        """Check if violations were detected"""
        text_lower = text.lower()
        
        # Keywords indicating violations
        violation_keywords = [
            "yes", "detected", "evidence", "visible", "observed",
            "mining", "excavation", "encroachment", "clearing",
            "unauthorized", "illegal", "violation"
        ]
        
        no_violation_keywords = [
            "no evidence", "not detected", "no violations",
            "normal", "natural", "no signs", "unclear"
        ]
        
        # Check for negative indicators first
        for keyword in no_violation_keywords:
            if keyword in text_lower:
                return False
        
        # Check for positive indicators
        for keyword in violation_keywords:
            if keyword in text_lower:
                return True
        
        return None  # Uncertain
    
    def _extract_summary(self, text):
        """Extract first 2-3 sentences as summary"""
        sentences = text.split('.')
        summary = '. '.join(sentences[:3]).strip()
        return summary if summary else text[:200]
    
    def _extract_confidence(self, text):
        """Extract confidence level"""
        text_lower = text.lower()
        
        if "high confidence" in text_lower or "confident" in text_lower:
            return "High"
        elif "medium confidence" in text_lower or "moderate" in text_lower:
            return "Medium"
        elif "low confidence" in text_lower or "uncertain" in text_lower:
            return "Low"
        else:
            return "Medium"  # Default
    
    def _extract_severity(self, text):
        """Extract severity assessment"""
        text_lower = text.lower()
        
        if "severe" in text_lower or "significant" in text_lower or "major" in text_lower:
            return "Severe"
        elif "moderate" in text_lower or "medium" in text_lower:
            return "Moderate"
        elif "minor" in text_lower or "small" in text_lower or "limited" in text_lower:
            return "Minor"
        else:
            return "Unknown"
    
    def _extract_location(self, text):
        """Extract location description"""
        text_lower = text.lower()
        
        # Common location patterns
        location_keywords = [
            "northeastern", "northwestern", "southeastern", "southwestern",
            "northern", "southern", "eastern", "western",
            "center", "central", "middle",
            "riverbank", "riverside", "along the river",
            "upper", "lower", "left", "right"
        ]
        
        for keyword in location_keywords:
            if keyword in text_lower:
                # Find sentence containing location
                for sentence in text.split('.'):
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return "Location not specified"
    
    def _extract_recommendations(self, text):
        """Extract recommendations if present"""
        text_lower = text.lower()
        
        if "recommend" in text_lower or "suggest" in text_lower:
            # Find recommendation sentences
            recommendations = []
            for sentence in text.split('.'):
                if "recommend" in sentence.lower() or "suggest" in sentence.lower():
                    recommendations.append(sentence.strip())
            return recommendations if recommendations else ["Field verification recommended"]
        
        return ["Field verification recommended"]