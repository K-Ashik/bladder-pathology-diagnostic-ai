"""
LLM Reasoner Module
Uses Groq Cloud to generate clinical reasoning explanations
Optimized for Python 3.11+
"""

import tomllib
from groq import Groq
import streamlit as st
import os

class ClinicalReasoner:
    def __init__(self):
        """Initialize Groq client with secrets from Streamlit Cloud"""
        
        # Try to get API key from Streamlit secrets first (for Streamlit Cloud)
        try:
            api_key = st.secrets["groq"]["api_key"]
        except (KeyError, FileNotFoundError):
            # Fall back to environment variable (for local development)
            api_key = os.getenv("GROQ_API_KEY")
        
        # If running locally, try to read from secrets.toml
        if not api_key:
            try:
                with open("secrets.toml", "rb") as f:
                    secrets = tomllib.load(f)
                api_key = secrets["groq"]["api_key"]
            except FileNotFoundError:
                raise FileNotFoundError("❌ GROQ API key not found. Please set it in Streamlit secrets or secrets.toml")
        
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("❌ Please set your GROQ_API_KEY in Streamlit secrets or secrets.toml")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        
        print("✅ Groq client initialized successfully")
        print(f"📊 Using model: {self.model}")
    
    def generate_clinical_report(self, prediction, confidence, selected_features, similar_cases_df=None):
        """
        Generate clinical reasoning report using LLM
        
        Args:
            prediction: Predicted diagnosis (e.g., 'malakoplakia')
            confidence: Confidence score (0-1)
            selected_features: List of selected morphological features
            similar_cases_df: DataFrame of similar cases (optional)
        
        Returns:
            Clinical reasoning report as string
        """
        
        # Prepare features text
        features_text = ", ".join(selected_features) if selected_features else "No specific features selected"
        
        # Prepare similar cases text
        similar_cases_text = ""
        if similar_cases_df is not None and len(similar_cases_df) > 0:
            similar_cases_text = "Similar cases from literature:\n"
            for idx, case in similar_cases_df.head(3).iterrows():
                similar_cases_text += f"- {case['title']}\n"
        
        # Create prompt for LLM
        prompt = f"""You are an expert pathologist providing clinical reasoning for a bladder biopsy diagnosis.

PATIENT BIOPSY FINDINGS:
Morphological features present: {features_text}

AI MODEL PREDICTION:
Primary diagnosis: {prediction.upper()}
Confidence score: {confidence:.1%}

SIMILAR CASES FROM LITERATURE:
{similar_cases_text if similar_cases_text else "No similar cases found in literature"}

Please provide a clinical reasoning report that includes:

1. **Clinical Interpretation**: Explain what these morphological features mean and why they support the diagnosis of {prediction}

2. **Differential Diagnosis**: List 2-3 other conditions that could present similarly and explain why they are less likely given these features

3. **Diagnostic Confidence**: Assess the confidence level of this diagnosis based on the features present

4. **Recommended Next Steps**: Suggest 2-3 specific tests or investigations to confirm this diagnosis (e.g., immunohistochemistry, special stains, cultures, imaging)

5. **Clinical Significance**: Briefly explain what this diagnosis means for patient management and treatment

6. **Literature Support**: Reference the similar cases found in medical literature that support this diagnosis

Format your response in clear sections with headers. Be concise but thorough. Use language appropriate for a pathologist."""

        try:
            # Call Groq API using chat completion
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            return f"❌ Error generating clinical report: {str(e)}"
    
    def generate_differential_diagnosis(self, prediction, selected_features):
        """
        Generate differential diagnosis analysis
        
        Args:
            prediction: Predicted diagnosis
            selected_features: List of morphological features
        
        Returns:
            Differential diagnosis analysis
        """
        
        features_text = ", ".join(selected_features) if selected_features else "No specific features"
        
        prompt = f"""As a pathologist, analyze the differential diagnosis for these bladder biopsy findings:

Features present: {features_text}
Primary prediction: {prediction}

Provide a brief differential diagnosis table with:
- Diagnosis name
- Likelihood (High/Medium/Low)
- Key distinguishing features
- Why ruled in or out

Keep it concise and practical."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.7
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            return f"❌ Error generating differential diagnosis: {str(e)}"
    
    def generate_summary(self, prediction, confidence):
        """
        Generate a one-line clinical summary for pathology report
        
        Args:
            prediction: Predicted diagnosis
            confidence: Confidence score (0-1)
        
        Returns:
            One-sentence clinical summary
        """
        
        prompt = f"""Provide a one-sentence clinical summary for a pathology report:
Diagnosis: {prediction}
Confidence: {confidence:.1%}

The summary should be concise, professional, and suitable for a pathology report."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            return f"{prediction.upper()} - confidence {confidence:.1%}"