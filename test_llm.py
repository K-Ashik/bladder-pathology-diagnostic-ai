"""
Test LLM Reasoner
Quick test to verify Groq integration works
"""

from llm_reasoner import ClinicalReasoner

print("=" * 80)
print("🧪 Testing LLM Reasoner")
print("=" * 80)
print()

try:
    # Initialize reasoner
    print("Initializing ClinicalReasoner...")
    reasoner = ClinicalReasoner()
    print()
    
    # Test data
    prediction = "malakoplakia"
    confidence = 0.75
    features = ["michaelis_gutmann", "foam_cells", "granuloma"]
    
    # Test 1: Generate summary
    print("Test 1: Generating one-line summary...")
    summary = reasoner.generate_summary(prediction, confidence)
    print(f"Summary: {summary}")
    print()
    
    # Test 2: Generate differential diagnosis
    print("Test 2: Generating differential diagnosis...")
    diff_diag = reasoner.generate_differential_diagnosis(prediction, features)
    print(diff_diag)
    print()
    
    # Test 3: Generate full clinical report
    print("Test 3: Generating full clinical report...")
    print("(This may take 5-10 seconds...)")
    report = reasoner.generate_clinical_report(prediction, confidence, features)
    print(report)
    print()
    
    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Check secrets.toml exists and has correct API key")
    print("2. Verify you replaced 'your_api_key_here' with actual key")
    print("3. Check internet connection")
    print("4. Verify Groq API key is valid on console.groq.com")