"""
Diagnostic AI Dashboard - Enhanced Version
Interactive Streamlit app for pathologists with patient tracking and visualizations
Button-controlled report generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_reasoner import ClinicalReasoner
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Pathology Diagnostic AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .patient-header {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1976d2;
    }
    .ready-banner {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .warning-banner {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    with open("best_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open("scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open("embeddings.pkl", 'rb') as f:
        embeddings = pickle.load(f)
    reasoner = ClinicalReasoner()
    return model, scaler, embeddings, reasoner

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_with_features.csv")
    metadata = pd.read_csv("embeddings_metadata.csv")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return df, metadata, embedding_model

# Load everything
model, scaler, embeddings, reasoner = load_models()
df, metadata, embedding_model = load_data()

# Get feature columns
feature_cols = [col for col in df.columns if col not in 
                ['pmid', 'title', 'abstract', 'year', 'journal', 'first_author', 
                 'num_authors', 'keywords', 'source_query', 'category', 'combined_text']]

# Initialize session state
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = ""
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = None
if 'cached_features' not in st.session_state:
    st.session_state.cached_features = {}

# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<h1 class="main-header">🔬 Bladder Pathology Diagnostic AI</h1>', 
            unsafe_allow_html=True)

st.markdown("""
**AI-Powered Diagnostic Assistant for Pathologists**

This tool helps pathologists with differential diagnosis of bladder lesions using:
- 🤖 Machine Learning prediction (76% accuracy on 463 research articles)
- 🧠 LLM clinical reasoning (Groq Llama 3.3 70B)
- 📚 Similar case retrieval from medical literature
- 📊 Visual analysis of morphological findings
- 📋 Automated clinical report generation with patient tracking

**Disclaimer:** This is a supporting tool, not a replacement for expert judgment.
""")

st.divider()

# ============================================================================
# SIDEBAR - PATIENT INFO & FEATURE INPUT
# ============================================================================

with st.sidebar:
    st.header("👤 Patient Information")
    
    # Patient ID input
    patient_id = st.text_input(
        "Patient ID *",
        value=st.session_state.patient_id,
        placeholder="e.g., PT-2025-001",
        help="Unique identifier for this patient"
    )
    
    st.session_state.patient_id = patient_id
    
    # Reset button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset All", use_container_width=True, key="reset_all"):
            st.session_state.patient_id = ""
            st.session_state.report_generated = False
            st.session_state.cached_results = None
            st.session_state.cached_features = {}
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear ID", use_container_width=True, key="clear_id"):
            st.session_state.patient_id = ""
            st.rerun()
    
    st.divider()
    
    st.header("🔍 Patient Morphological Features")
    st.markdown("Select features present in the biopsy:")
    
    # Create input fields for each feature
    feature_inputs = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_inputs['michaelis_gutmann'] = st.checkbox("Michaelis-Gutmann Bodies")
        feature_inputs['von_hansemann'] = st.checkbox("Von Hansemann Cells")
        feature_inputs['granuloma'] = st.checkbox("Granuloma")
        feature_inputs['foam_cells'] = st.checkbox("Foam Cells")
        feature_inputs['calcification'] = st.checkbox("Calcification")
        feature_inputs['egg'] = st.checkbox("Parasitic Eggs")
        feature_inputs['terminal_spine'] = st.checkbox("Terminal Spine")
        feature_inputs['lamination'] = st.checkbox("Lamination")
    
    with col2:
        feature_inputs['inflammation'] = st.checkbox("Inflammation")
        feature_inputs['ulceration'] = st.checkbox("Ulceration")
        feature_inputs['fibrosis'] = st.checkbox("Fibrosis")
        feature_inputs['necrosis'] = st.checkbox("Necrosis")
        feature_inputs['cystitis'] = st.checkbox("Cystitis")
        feature_inputs['eosinophil'] = st.checkbox("Eosinophil Infiltration")
        feature_inputs['central_material'] = st.checkbox("Central Material")
    
    st.divider()
    
    # Convert to binary array
    X_input = np.array([feature_inputs[col] for col in feature_cols]).reshape(1, -1)
    
    # Scale
    X_scaled = scaler.transform(X_input)
    
    # Get prediction
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Get class names
    class_names = model.classes_
    
    # Get confidence
    max_prob_idx = np.argmax(probabilities)
    confidence = probabilities[max_prob_idx]
    
    st.header("📋 ML Prediction (Quick Preview)")
    st.info(f"**Primary: {prediction.upper()}** | **Confidence: {confidence:.1%}**")
    
    # Feature count
    selected_features = [col for col, val in feature_inputs.items() if val]
    
    st.divider()
    
    # Check if ready to generate report
    is_ready = bool(st.session_state.patient_id) and len(selected_features) > 0
    
    if is_ready:
        st.markdown("""
        <div class="ready-banner">
        ✅ <strong>Ready to Generate Report!</strong><br/>
        Patient ID and morphological features are complete.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-banner">
        ⚠️ <strong>Not Ready Yet</strong><br/>
        """, unsafe_allow_html=True)
        if not st.session_state.patient_id:
            st.markdown("• Please enter a Patient ID", unsafe_allow_html=True)
        if len(selected_features) == 0:
            st.markdown("• Please select at least one morphological feature", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # MAIN GENERATE REPORT BUTTON
    generate_button = st.button(
        "🚀 GENERATE FULL REPORT",
        use_container_width=True,
        disabled=not is_ready,
        key="generate_report"
    )
    
    if generate_button:
        st.session_state.report_generated = True
        # Cache the results
        st.session_state.cached_results = {
            'prediction': prediction,
            'probabilities': probabilities,
            'class_names': class_names,
            'confidence': confidence,
            'selected_features': selected_features,
            'feature_inputs': feature_inputs
        }
        st.rerun()
    
    st.divider()
    st.metric(label="Features Selected", value=len(selected_features))

# ============================================================================
# MAIN CONTENT - RESULTS (Only if report is generated)
# ============================================================================

# Show patient header if report is generated
if st.session_state.report_generated and st.session_state.cached_results:
    st.markdown(f"""
    <div class="patient-header">
    <h3>👤 Patient Report</h3>
    <p><strong>Patient ID:</strong> {st.session_state.patient_id}</p>
    <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Morphological Features Selected:</strong> {len(st.session_state.cached_results['selected_features'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get cached results
    prediction = st.session_state.cached_results['prediction']
    confidence = st.session_state.cached_results['confidence']
    probabilities = st.session_state.cached_results['probabilities']
    class_names = st.session_state.cached_results['class_names']
    selected_features = st.session_state.cached_results['selected_features']
    feature_inputs = st.session_state.cached_results['feature_inputs']
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Clinical Report", "📊 Visual Analysis", "🔍 Diagnosis", "📚 Similar Cases", "📈 Info"])
    
    with tab1:
        st.header("🧠 LLM Clinical Reasoning Report")
        
        # Find similar cases
        if selected_features:
            feature_text = "Features: " + ", ".join(selected_features)
        else:
            feature_text = "No specific features selected"
        
        user_embedding = embedding_model.encode([feature_text])[0]
        similarities = cosine_similarity([user_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        similar_cases = metadata.iloc[top_indices]
        
        # Generate clinical report
        st.info("🔄 Generating clinical reasoning... (using Groq Llama 3.3 70B)")
        
        with st.spinner("Generating comprehensive clinical report..."):
            clinical_report = reasoner.generate_clinical_report(
                prediction=prediction,
                confidence=confidence,
                selected_features=selected_features,
                similar_cases_df=similar_cases
            )
        
        # Display report in formatted sections
        st.markdown(clinical_report)
        
        st.divider()
        
        # Create downloadable report with patient info
        full_report = f"""================================================================================
PATHOLOGY DIAGNOSTIC REPORT
================================================================================

PATIENT INFORMATION:
Patient ID: {st.session_state.patient_id}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
AI PREDICTION SUMMARY:
Primary Diagnosis: {prediction.upper()}
Confidence Level: {confidence:.1%}

Selected Morphological Features ({len(selected_features)} features):
{chr(10).join([f"  • {f.replace('_', ' ').title()}" for f in selected_features])}

================================================================================
CLINICAL REASONING:
{clinical_report}

================================================================================
DISCLAIMER:
This report is generated using AI and should be used as a supporting tool only.
Professional pathologist review and judgment are essential for clinical diagnosis.
================================================================================
"""
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download as TXT",
                data=full_report,
                file_name=f"pathology_report_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Create CSV version
            report_csv = pd.DataFrame({
                'Field': [
                    'Patient ID',
                    'Report Date',
                    'Primary Diagnosis',
                    'Confidence',
                    'Features Selected',
                    'Number of Features'
                ],
                'Value': [
                    st.session_state.patient_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    prediction.upper(),
                    f"{confidence:.1%}",
                    ", ".join([f.replace('_', ' ').title() for f in selected_features]) if selected_features else "None",
                    len(selected_features)
                ]
            })
            
            st.download_button(
                label="📊 Download as CSV",
                data=report_csv.to_csv(index=False),
                file_name=f"pathology_report_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("↩️ Back to Input", use_container_width=True, key="back_button"):
                st.session_state.report_generated = False
                st.session_state.cached_results = None
                st.rerun()
    
    with tab2:
        st.header("📊 Visual Analysis of Morphological Findings")
        
        # 1. Feature Presence Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Feature Presence")
            
            # Bar chart of selected features
            feature_names = [f.replace('_', ' ').title() for f in selected_features]
            fig = px.bar(
                x=feature_names,
                y=[1]*len(selected_features),
                labels={'x': 'Morphological Features', 'y': 'Present'},
                color=[1]*len(selected_features),
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                showlegend=False,
                yaxis={'visible': False}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Diagnosis Probability Distribution")
            
            # Pie chart of diagnosis probabilities
            fig = go.Figure(data=[go.Pie(
                labels=class_names,
                values=probabilities,
                hole=.3,
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 2. Feature Categories Analysis
        st.subheader("📈 Feature Category Breakdown")
        
        # Group features by category
        feature_categories = {
            'Structural': ['michaelis_gutmann', 'von_hansemann', 'central_material', 'lamination'],
            'Inflammatory': ['granuloma', 'inflammation', 'eosinophil'],
            'Parasitic': ['egg', 'terminal_spine'],
            'Degenerative': ['calcification', 'fibrosis', 'ulceration', 'necrosis'],
            'Infection': ['cystitis', 'foam_cells']
        }
        
        category_data = {}
        for category, features in feature_categories.items():
            count = sum(1 for f in features if feature_inputs.get(f, False))
            category_data[category] = count
        
        fig = px.bar(
            x=list(category_data.keys()),
            y=list(category_data.values()),
            labels={'x': 'Feature Category', 'y': 'Features Present'},
            color=list(category_data.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 3. Confidence Meter
        st.subheader("🎯 Diagnostic Confidence Level")
        
        fig = go.Figure(data=[
            go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Score (%)"},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "#ff6b6b"},
                        {'range': [25, 50], 'color': "#ffa500"},
                        {'range': [50, 75], 'color': "#ffd700"},
                        {'range': [75, 100], 'color': "#51cf66"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 4. Summary Statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Features Selected", len(selected_features))
        
        with col2:
            st.metric("Primary Diagnosis", prediction.replace('_', ' ').title())
        
        with col3:
            st.metric("Confidence Score", f"{confidence:.1%}")
        
        with col4:
            if confidence >= 0.75:
                st.metric("Risk Level", "🟢 Low")
            elif confidence >= 0.60:
                st.metric("Risk Level", "🟡 Medium")
            else:
                st.metric("Risk Level", "🔴 High")
    
    with tab3:
        st.header("📊 Diagnosis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Primary Diagnosis**\n\n{prediction.upper()}")
        
        with col2:
            st.info(f"**ML Confidence**\n\n{confidence:.1%}")
        
        with col3:
            if confidence >= 0.75:
                st.success(f"**Confidence Level**\n\n✅ High")
            elif confidence >= 0.60:
                st.warning(f"**Confidence Level**\n\n⚠️ Medium")
            else:
                st.error(f"**Confidence Level**\n\n❌ Low")
        
        st.divider()
        
        st.subheader("Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Features Present:**")
            if selected_features:
                for feature in selected_features:
                    st.write(f"• {feature.replace('_', ' ').title()}")
            else:
                st.write("No features selected")
        
        with col2:
            st.markdown("**Prediction Breakdown:**")
            for class_name, prob in sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True):
                st.write(f"• {class_name}: {prob:.1%}")
        
        st.divider()
        
        # Differential diagnosis from LLM
        st.subheader("🔍 Differential Diagnosis (LLM Analysis)")
        
        with st.spinner("Generating differential diagnosis..."):
            diff_diag = reasoner.generate_differential_diagnosis(prediction, selected_features)
        
        st.markdown(diff_diag)
    
    with tab4:
        st.header("📚 Similar Cases from Literature")
        
        # Get selected features
        selected_features_list = selected_features
        
        # Create embedding for user input
        if selected_features_list:
            feature_text = "Features: " + ", ".join(selected_features_list)
        else:
            feature_text = "No specific features selected"
        
        user_embedding = embedding_model.encode([feature_text])[0]
        
        # Find similar cases
        similarities = cosine_similarity([user_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:10]
        
        st.markdown(f"Found **{len(top_indices)} most similar cases** from 463 research articles:")
        st.divider()
        
        for rank, idx in enumerate(top_indices, 1):
            row = metadata.iloc[idx]
            
            col1, col2 = st.columns([1, 20])
            with col1:
                st.write(f"**#{rank}**")
            with col2:
                with st.expander(f"PMID: {int(row['pmid'])} | {row['category'].upper()} | Similarity: {similarities[idx]:.1%}"):
                    st.markdown(f"**Title:** {row['title']}")
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Similarity Score:** {similarities[idx]:.1%}")
                    st.markdown(f"**PubMed Link:** https://pubmed.ncbi.nlm.nih.gov/{int(row['pmid'])}/")
    
    with tab5:
        st.header("📊 Model & System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"**Total Articles:** 463")
            st.write(f"**Training Articles:** 370")
            st.write(f"**Test Articles:** 93")
            st.write(f"**Features:** 15 morphological features")
            st.write(f"**Categories:** 5 diagnostic categories")
        
        with col2:
            st.subheader("ML Model Performance")
            st.write(f"**Test Accuracy:** 76.3%")
            st.write(f"**Cross-Validation Accuracy:** 74.6% (±4.0%)")
            st.write(f"**Algorithm:** Logistic Regression")
            st.write(f"**Scaler:** StandardScaler")
        
        st.divider()
        
        st.subheader("LLM System")
        st.write(f"**Provider:** Groq Cloud")
        st.write(f"**Model:** Llama 3.3 70B Versatile")
        st.write(f"**Inference Speed:** <2 seconds")
        st.write(f"**Purpose:** Clinical reasoning & explanations")
        
        st.divider()
        
        st.subheader("Diagnostic Categories Distribution")
        category_dist = df['category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            for cat, count in category_dist.items():
                pct = (count / len(df)) * 100
                st.write(f"• {cat.replace('_', ' ').title()}: {count} articles ({pct:.1f}%)")
        
        with col2:
            st.bar_chart(category_dist)
        
        st.divider()
        
        st.markdown("""
        ### How to Use This Tool
        
        1. **Enter Patient ID:** Start by entering a unique patient identifier
        2. **Select Features:** In the left sidebar, check the morphological features present in your biopsy
        3. **Generate Report:** Click the "🚀 GENERATE FULL REPORT" button when ready
        4. **Review Results:**
           - Clinical reasoning from LLM
           - Visual charts showing findings
           - Diagnostic predictions
           - Similar cases from literature
        5. **Download Report:** Export as TXT or CSV with patient ID
        6. **Reset:** Use the "🔄 Reset All" button to start with a new patient
        
        ### Important Notes
        - Patient ID is used to track and identify reports
        - Report generation only starts after clicking the button
        - This tool provides **supporting evidence**, not definitive diagnosis
        - Always use professional judgment and clinical context
        - Correlate with imaging, clinical history, and other findings
        - For complex cases, consult with specialist pathologists
        
        ### Technology Stack
        - **Data:** 463 PubMed research articles
        - **ML:** Scikit-learn Logistic Regression (76% accuracy)
        - **LLM:** Groq Cloud (Llama 3.3 70B)
        - **RAG:** Sentence Transformers for semantic search
        - **Visualization:** Plotly for interactive charts
        - **UI:** Streamlit
        """)

else:
    # Show welcome message if no report generated
    st.info("""
    ### 👋 Welcome to Bladder Pathology Diagnostic AI
    
    **Get started:**
    1. Enter a **Patient ID** in the left sidebar
    2. Select the **morphological features** you observed in the biopsy
    3. Click the **"🚀 GENERATE FULL REPORT"** button when ready
    
    The system will then:
    - ✅ Generate AI clinical reasoning
    - ✅ Show visual charts of your findings
    - ✅ Retrieve similar cases from literature
    - ✅ Create a downloadable report with patient tracking
    
    **Note:** Report generation only starts after you click the button.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**Bladder Pathology Diagnostic AI** | Built with ❤️ using Streamlit + Groq LLM
Data: 463 PubMed articles | ML Accuracy: 76.3% | LLM: Llama 3.3 70B

⚠️ **Disclaimer:** For research and educational purposes only. Not for clinical use without expert review.
""")