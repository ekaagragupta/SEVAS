"""
Test Vision AI integration
"""

from models.vision_ai import VisionAI
import os

# Initialize
vision_ai = VisionAI()

# Test image
test_image_path = 'uploads/test_image.jpg'

if not os.path.exists(test_image_path):
    print(" Test image not found")
else:
    print("=" * 70)
    print(" TESTING VISION AI - SEVAS")
    print("=" * 70)
    
    # Test 1: General analysis
    print("\n" + "="*70)
    print("TEST 1: General Environmental Analysis")
    print("="*70)
    
    result = vision_ai.analyze_with_gemini(test_image_path, detection_type="general")
    
    if "error" not in result or result["error"] is None:
        print(f"\n Analysis Results:")
        print(f"   Violations detected: {result.get('violations_detected', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 'Unknown')}")
        print(f"   Severity: {result.get('severity', 'Unknown')}")
        print(f"   Location: {result.get('location', 'Not specified')}")
        
        print(f"\n Summary:")
        print(f"   {result.get('summary', 'No summary available')}")
        
        print(f"\n Full AI Response:")
        print("-" * 70)
        print(result.get('raw_response', 'No response'))
        print("-" * 70)
    else:
        print(f" Error: {result['error']}")
    
    # Test 2: Sand mining specific
    print("\n" + "="*70)
    print("TEST 2: Sand Mining Detection")
    print("="*70)
    
    result = vision_ai.analyze_with_gemini(test_image_path, detection_type="sand_mining")
    
    if "error" not in result or result["error"] is None:
        print(f"\n Sand Mining Analysis:")
        print(f"   Evidence found: {result.get('violations_detected', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 'Unknown')}")
        
        print(f"\n💬 AI Assessment:")
        print("-" * 70)
        print(result.get('raw_response', 'No response'))
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print(" VISION AI TESTING COMPLETE!")
    print("=" * 70)