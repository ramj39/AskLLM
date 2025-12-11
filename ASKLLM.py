import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
import pandas as pd
import re
import requests
import base64
import json
from io import BytesIO
from typing import Dict, List, Optional

# ----------------------------
# LLM CONFIGURATION
# ----------------------------
LLM_CONFIG = {
    "api_url": "https://api.groq.com/openai/v1/chat/completions",
    "default_model": "llama-3.1-8b-instant",
    "api_key": ""
}

# Initialize session states
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

if "chemistry_results" not in st.session_state:
    st.session_state.chemistry_results = {}

# ----------------------------
# CHEMISTRY FUNCTIONS
# ----------------------------

def get_compound_database():
    """Database of common compounds with SMILES"""
    return {
        # Alcohols
        'benzyl alcohol': 'c1ccccc1CO',
        'ethanol': 'CCO',
        'methanol': 'CO',
        'cyclohexanol': 'OC1CCCCC1',
        'isopropanol': 'CC(C)O',
        
        # Aldehydes
        'benzaldehyde': 'c1ccccc1C=O',
        'acetaldehyde': 'CC=O',
        'formaldehyde': 'C=O',
        
        # Carboxylic acids
        'benzoic acid': 'c1ccccc1C(=O)O',
        'acetic acid': 'CC(=O)O',
        'formic acid': 'OC=O',
        'oxalic acid': 'OC(=O)C(=O)O',
        'malonic acid': 'OC(=O)CC(=O)O',
        
        # Ketones
        'acetophenone': 'CC(=O)c1ccccc1',
        'acetone': 'CC(=O)C',
        'cyclohexanone': 'O=C1CCCCC1',
        
        # Aromatic compounds
        'toluene': 'Cc1ccccc1',
        'benzene': 'c1ccccc1',
        'phenol': 'Oc1ccccc1',
        'aniline': 'Nc1ccccc1',
        'nitrobenzene': 'O=[N+]([O-])c1ccccc1',
        'bromobenzene': 'Brc1ccccc1',
        
        # Esters and derivatives
        'methyl benzoate': 'COC(=O)c1ccccc1',
        'ethyl acetate': 'CCOC(=O)C',
        'acetanilide': 'CC(=O)Nc1ccccc1',
        
        # Amides and amines
        'ammonia': 'N',
        'oxamide': 'NC(=O)C(=O)N',
    }

def get_reaction_pathways():
    """All available reaction pathways"""
    return {
        'oxidation': [
            {
                'name': 'Primary Alcohol Oxidation',
                'reactants': ['c1ccccc1CO'],
                'products': ['c1ccccc1C(=O)O'],
                'description': 'Primary alcohol → Aldehyde → Carboxylic acid',
                'reagents': ['KMnO₄', 'K₂Cr₂O₇/H₂SO₄'],
                'mechanism': 'Stepwise oxidation via chromate ester intermediate'
            }
        ],
        'reduction': [
            {
                'name': 'Nitro Reduction',
                'reactants': ['O=[N+]([O-])c1ccccc1'],
                'products': ['Nc1ccccc1'],
                'description': 'Nitro group → Amino group',
                'reagents': ['Sn/HCl', 'H₂/Pd'],
                'mechanism': 'Stepwise reduction via nitroso intermediate'
            }
        ],
        'esterification': [
            {
                'name': 'Fischer Esterification',
                'reactants': ['c1ccccc1C(=O)O', 'CO'],
                'products': ['COC(=O)c1ccccc1'],
                'description': 'Carboxylic acid + Alcohol → Ester',
                'reagents': ['H₂SO₄ (cat.)'],
                'mechanism': 'Nucleophilic acyl substitution with acid catalysis'
            }
        ],
        'ammonolysis': [
            {
                'name': 'Oxalic Acid to Oxamide',
                'reactants': ['OC(=O)C(=O)O', 'N'],
                'products': ['NC(=O)C(=O)N'],
                'description': 'Oxalic acid + Ammonia → Oxamide',
                'reagents': ['NH₃ (conc.)', 'Heat'],
                'mechanism': 'Nucleophilic acyl substitution'
            }
        ]
    }

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def get_compound_name(smiles: str) -> str:
    """Get compound name from SMILES"""
    database = get_compound_database()
    for name, smi in database.items():
        if smi == smiles:
            return name.title()
    return "Unknown compound"

def draw_molecule(smiles: str, size=(200, 200)):
    """Draw a molecule from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Draw.MolToImage(mol, size=size)
    except:
        return None
    return None

def calculate_molecular_properties(smiles: str) -> Dict:
    """Calculate comprehensive molecular properties"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Calculate all properties
            props = {
                'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} g/mol",
                'Molecular Formula': rdMolDescriptors.CalcMolFormula(mol),
                'Exact Mass': f"{Descriptors.ExactMolWt(mol):.4f}",
                'Heavy Atom Count': mol.GetNumHeavyAtoms(),
                'Atom Count': mol.GetNumAtoms(),
                'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                'Hydrogen Bond Donors': Descriptors.NumHDonors(mol),
                'Hydrogen Bond Acceptors': Descriptors.NumHAcceptors(mol),
                'LogP (Octanol-Water)': f"{Descriptors.MolLogP(mol):.2f}",
                'TPSA': f"{Descriptors.TPSA(mol):.2f} Å²",
                'Molar Refractivity': f"{Descriptors.MolMR(mol):.2f}",
                'Ring Count': Descriptors.RingCount(mol),
                'Aromatic Rings': Descriptors.NumAromaticRings(mol),
                'Saturated Rings': Descriptors.NumSaturatedRings(mol),
                'Fraction CSP3': f"{Descriptors.FractionCSP3(mol):.3f}",
                'Number of Radical Electrons': Descriptors.NumRadicalElectrons(mol),
                'Number of Valence Electrons': Descriptors.NumValenceElectrons(mol),
                'Number of Heteroatoms': Descriptors.NumHeteroatoms(mol)
            }
            
            # Try to calculate additional properties
            try:
                props['Formal Charge'] = Chem.GetFormalCharge(mol)
            except:
                props['Formal Charge'] = 'N/A'
            
            try:
                props['Num. Aliphatic Carbocycles'] = Descriptors.NumAliphaticCarbocycles(mol)
                props['Num. Aliphatic Heterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
                props['Num. Aliphatic Rings'] = Descriptors.NumAliphaticRings(mol)
                props['Num. Aromatic Carbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
                props['Num. Aromatic Heterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
            except:
                pass
            
            return props
    except Exception as e:
        st.error(f"Error calculating properties: {str(e)}")
    return {}

def display_molecular_properties(smiles: str, compound_name: str = None):
    """Display molecular properties in a nice format"""
    if not validate_smiles(smiles):
        st.warning("Invalid SMILES - cannot calculate properties")
        return
    
    props = calculate_molecular_properties(smiles)
    
    if not props:
        st.info("No properties could be calculated")
        return
    
    # Create expander for properties
    with st.expander(f"📊 Molecular Properties: {compound_name or get_compound_name(smiles)}", expanded=False):
        
        # Display in columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Properties**")
            st.markdown(f"- **Molecular Formula:** {props.get('Molecular Formula', 'N/A')}")
            st.markdown(f"- **Molecular Weight:** {props.get('Molecular Weight', 'N/A')}")
            st.markdown(f"- **Exact Mass:** {props.get('Exact Mass', 'N/A')}")
            st.markdown(f"- **Atom Count:** {props.get('Atom Count', 'N/A')}")
            st.markdown(f"- **Heavy Atoms:** {props.get('Heavy Atom Count', 'N/A')}")
            st.markdown(f"- **Formal Charge:** {props.get('Formal Charge', 'N/A')}")
        
        with col2:
            st.markdown("**Physicochemical Properties**")
            st.markdown(f"- **LogP:** {props.get('LogP (Octanol-Water)', 'N/A')}")
            st.markdown(f"- **TPSA:** {props.get('TPSA', 'N/A')}")
            st.markdown(f"- **Molar Refractivity:** {props.get('Molar Refractivity', 'N/A')}")
            st.markdown(f"- **H-Bond Donors:** {props.get('Hydrogen Bond Donors', 'N/A')}")
            st.markdown(f"- **H-Bond Acceptors:** {props.get('Hydrogen Bond Acceptors', 'N/A')}")
            st.markdown(f"- **Rotatable Bonds:** {props.get('Rotatable Bonds', 'N/A')}")
        
        # Structural properties in a separate section
        st.markdown("**Structural Features**")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown(f"- **Total Rings:** {props.get('Ring Count', 'N/A')}")
            st.markdown(f"- **Aromatic Rings:** {props.get('Aromatic Rings', 'N/A')}")
            st.markdown(f"- **Saturated Rings:** {props.get('Saturated Rings', 'N/A')}")
            st.markdown(f"- **Fraction CSP3:** {props.get('Fraction CSP3', 'N/A')}")
        
        with col4:
            st.markdown(f"- **Heteroatoms:** {props.get('Number of Heteroatoms', 'N/A')}")
            st.markdown(f"- **Valence Electrons:** {props.get('Number of Valence Electrons', 'N/A')}")
            st.markdown(f"- **Radical Electrons:** {props.get('Number of Radical Electrons', 'N/A')}")
        
        # Optional: Show as table too
        if st.checkbox("Show all properties as table", key=f"table_{smiles}"):
            df = pd.DataFrame(list(props.items()), columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True)

def predict_product(reactant_smiles: str, reaction_type: str) -> Optional[str]:
    """Predict product based on reactant and reaction type"""
    prediction_rules = {
        'oxidation': {
            'c1ccccc1CO': 'c1ccccc1C=O',
            'CCO': 'CC=O',
        },
        'reduction': {
            'O=[N+]([O-])c1ccccc1': 'Nc1ccccc1',
        },
        'esterification': {
            'c1ccccc1C(=O)O': 'COC(=O)c1ccccc1',
        },
        'ammonolysis': {
            'OC(=O)C(=O)O': 'NC(=O)C(=O)N',
        }
    }
    
    if reaction_type in prediction_rules:
        return prediction_rules[reaction_type].get(reactant_smiles)
    return None

def predict_products_from_multiple(reactants: List[str], reaction_type: str) -> List[str]:
    """Predict products from multiple reactants"""
    predictions = []
    for reactant in reactants:
        predicted = predict_product(reactant, reaction_type)
        if predicted and predicted not in predictions:
            predictions.append(predicted)
    return predictions

def find_matching_pathways(reactants: List[str], reaction_type: str) -> List[Dict]:
    """Find pathways matching the reactants"""
    pathways = get_reaction_pathways()
    matching = []
    
    if reaction_type in pathways:
        for pathway in pathways[reaction_type]:
            # Simple matching logic
            matching.append(pathway)
    
    return matching

# ----------------------------
# LLM FUNCTIONS
# ----------------------------

def call_llm(prompt: str, api_key: str = None, model: str = None) -> str:
    """Call LLM API"""
    if not api_key:
        return "Please enter your API key in the sidebar"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model or LLM_CONFIG["default_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(LLM_CONFIG["api_url"], headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# MAIN APP
# ----------------------------

def main():
    st.set_page_config(
        page_title="Chemistry AI Assistant",
        page_icon="🧪",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🧪 Chemistry AI Assistant")
        
        # Mode selection
        app_mode = st.radio(
            "Select Mode:",
            ["Chemistry Solver", "AI Assistant", "LLM Chat"]
        )
        
        st.markdown("---")
        
        # LLM Settings
        if app_mode in ["AI Assistant", "LLM Chat"]:
            st.subheader("🤖 LLM Settings")
            LLM_CONFIG["api_key"] = st.text_input("API Key", type="password")
            LLM_CONFIG["default_model"] = st.selectbox(
                "Model",
                ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
            )
        
        # Quick tools
        st.subheader("🔧 Quick Tools")
        with st.expander("SMILES Validator"):
            test_smiles = st.text_input("Test SMILES:", "c1ccccc1C(=O)O")
            if test_smiles:
                if validate_smiles(test_smiles):
                    st.success("✅ Valid")
                    img = draw_molecule(test_smiles)
                    if img:
                        st.image(img, caption=get_compound_name(test_smiles))
                    # Show properties
                    display_molecular_properties(test_smiles)
                else:
                    st.error("❌ Invalid")
    
    # Main content based on mode
    if app_mode == "Chemistry Solver":
        render_chemistry_solver()
    elif app_mode == "AI Assistant":
        render_ai_assistant()
    else:
        render_llm_chat()

def render_chemistry_solver():
    """Mode 1: Chemistry Solver with 5-reactant capability"""
    st.title("🧪 Chemistry Solver")
    st.markdown("Analyze reactions with 1-5 reactants")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        num_reactants = st.slider("Number of reactants:", 1, 5, 1)
        
        reaction_types = {
            "oxidation": "Increase O/decrease H",
            "reduction": "Decrease O/increase H",
            "esterification": "Acid + Alcohol → Ester",
            "ammonolysis": "Acid + NH₃ → Amide"
        }
        
        reaction_type = st.selectbox(
            "Reaction type:",
            list(reaction_types.keys()),
            format_func=lambda x: f"{x.title()} - {reaction_types[x]}"
        )
    
    with col2:
        st.info(f"**Setup:** {num_reactants} reactant(s), {reaction_type}")
    
    st.markdown("---")
    
    # Reactant input
    st.subheader("📦 Reactant Input")
    reactants = []
    
    # Create columns for input
    cols = st.columns(min(num_reactants, 3))
    
    for i in range(num_reactants):
        col_idx = i % 3
        with cols[col_idx]:
            # Smart defaults
            defaults = ['c1ccccc1CO', 'CC=O', 'CC(=O)O', 'N', 'Cc1ccccc1']
            default = defaults[i] if i < len(defaults) else ""
            
            r_smiles = st.text_input(
                f"Reactant {i+1} SMILES:",
                default,
                key=f"r{i}"
            )
            
            if r_smiles:
                reactants.append(r_smiles)
                
                # Validate and show
                if validate_smiles(r_smiles):
                    st.success(f"✅ {get_compound_name(r_smiles)}")
                    
                    # Show molecule
                    img = draw_molecule(r_smiles, (150, 150))
                    if img:
                        st.image(img, caption=f"R{i+1}")
                    
                    # Properties button
                    if st.button(f"📊 Properties R{i+1}", key=f"props_r{i}"):
                        display_molecular_properties(r_smiles, f"Reactant {i+1}")
                else:
                    st.error("❌ Invalid SMILES")
    
    # Analyze button
    if st.button("🔬 Analyze Reaction", type="primary"):
        if not reactants:
            st.warning("Please enter at least one reactant!")
        else:
            with st.spinner("Analyzing..."):
                
                # Display analysis
                st.subheader("🔬 Analysis Results")
                
                # 1. Show reactants with properties
                st.markdown("### 📦 Reactants Analysis")
                for i, r_smiles in enumerate(reactants):
                    if validate_smiles(r_smiles):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            img = draw_molecule(r_smiles, (200, 200))
                            if img:
                                st.image(img, caption=f"Reactant {i+1}")
                        with col2:
                            st.write(f"**Reactant {i+1}:** `{r_smiles}`")
                            st.write(f"**Name:** {get_compound_name(r_smiles)}")
                            # Display properties inline
                            display_molecular_properties(r_smiles)
                
                # 2. Predict products
                st.markdown("### 🎯 Predicted Products")
                predictions = predict_products_from_multiple(reactants, reaction_type)
                
                if predictions:
                    for i, p_smiles in enumerate(predictions):
                        if validate_smiles(p_smiles):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                img = draw_molecule(p_smiles, (200, 200))
                                if img:
                                    st.image(img, caption=f"Product {i+1}")
                            with col2:
                                st.write(f"**Product {i+1}:** `{p_smiles}`")
                                st.write(f"**Name:** {get_compound_name(p_smiles)}")
                                # Display product properties
                                display_molecular_properties(p_smiles)
                else:
                    st.info("No predictions available for this combination")
                
                # 3. Show pathways
                st.markdown("### 📚 Matching Pathways")
                pathways = find_matching_pathways(reactants, reaction_type)
                
                if pathways:
                    for pathway in pathways:
                        with st.expander(f"🔬 {pathway['name']}"):
                            st.write(f"**Description:** {pathway['description']}")
                            if 'reagents' in pathway:
                                st.write("**Reagents:** " + ", ".join(pathway['reagents']))
                            if 'mechanism' in pathway:
                                st.write("**Mechanism:**")
                                st.info(pathway['mechanism'])
                else:
                    st.info("No predefined pathways match")

def render_ai_assistant():
    """Mode 2: AI Assistant with chemistry focus"""
    st.title("🤖 AI Chemistry Assistant")
    
    if not LLM_CONFIG["api_key"]:
        st.warning("Please enter your API key in the sidebar")
        return
    
    # Two-column layout
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.subheader("🧪 Chemistry Query")
        
        # Option 1: Natural language
        query = st.text_area(
            "Describe your chemistry question:",
            height=150,
            placeholder="e.g., 'Explain the mechanism of esterification between benzoic acid and methanol'"
        )
        
        # Option 2: Structured input
        with st.expander("Structured Input"):
            struct_query = st.text_area(
                "Or enter SMILES and reaction type:",
                placeholder="Reactants: c1ccccc1C(=O)O, CO\nReaction: esterification\nQuestion: What's the mechanism?"
            )
        
        # Add molecular properties context
        with st.expander("Add Molecular Properties Context"):
            props_smiles = st.text_input("SMILES for property analysis:", "c1ccccc1C(=O)O")
            if props_smiles and validate_smiles(props_smiles):
                props = calculate_molecular_properties(props_smiles)
                if props:
                    st.write(f"**Properties for {get_compound_name(props_smiles)}:**")
                    st.write(f"- Formula: {props.get('Molecular Formula')}")
                    st.write(f"- MW: {props.get('Molecular Weight')}")
                    st.write(f"- LogP: {props.get('LogP (Octanol-Water)')}")
        
        if st.button("🤔 Ask AI", type="primary"):
            if query or struct_query:
                # Prepare prompt with chemistry context
                prompt = f"""As a chemistry expert, answer this question:

                QUESTION: {query if query else struct_query}
                
                Please provide:
                1. Detailed mechanism if applicable
                2. Step-by-step explanation
                3. Key reagents and conditions
                4. Any safety considerations
                5. Practical applications
                
                Use proper chemical terminology and be thorough."""
                
                # Call LLM
                with st.spinner("Consulting chemistry expert..."):
                    response = call_llm(prompt, LLM_CONFIG["api_key"], LLM_CONFIG["default_model"])
                    
                    # Store in session
                    st.session_state.llm_messages.append({
                        "role": "user",
                        "content": query or struct_query
                    })
                    st.session_state.llm_messages.append({
                        "role": "assistant",
                        "content": response
                    })
    
    with col_output:
        st.subheader("💬 AI Response")
        
        # Display conversation
        for msg in st.session_state.llm_messages:
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['content']}")
            
            st.markdown("---")
        
        # Download option
        if st.session_state.llm_messages:
            chat_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in st.session_state.llm_messages
            ])
            
            st.download_button(
                "📥 Download Conversation",
                chat_text,
                file_name="chemistry_chat.txt"
            )

def render_llm_chat():
    """Mode 3: General LLM Chat"""
    st.title("💬 General LLM Chat")
    
    if not LLM_CONFIG["api_key"]:
        st.warning("Please enter your API key in the sidebar")
        return
    
    # Chat interface
    user_input = st.text_area("Your message:", height=100)
    
    if st.button("📤 Send", type="primary"):
        if user_input:
            with st.spinner("Thinking..."):
                # Add user message
                st.session_state.llm_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Get response
                response = call_llm(
                    user_input,
                    LLM_CONFIG["api_key"],
                    LLM_CONFIG["default_model"]
                )
                
                # Add assistant response
                st.session_state.llm_messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.rerun()
    
    # Display chat
    st.markdown("---")
    st.subheader("💬 Conversation")
    
    for msg in st.session_state.llm_messages[-10:]:  # Show last 10 messages
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
        st.markdown("---")
st.markdown("[reference-wikipedia](https://en.wikipedia.org/wiki/List_of_organic_reactions)")
st.info("developed by subramanian ramajayam")
st.snow()
            
if __name__ == "__main__":
    main()
