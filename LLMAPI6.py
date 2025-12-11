import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
import pandas as pd
import re
import requests
import base64
import json
from io import BytesIO
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# ----------------------------
# LLM CONFIGURATION
# ----------------------------
LLM_CONFIG = {
    "api_url": "https://api.groq.com/openai/v1/chat/completions",
    "default_model": "llama-3.1-8b-instant",
    "api_key": ""  # Will be set by user
}

# Initialize LLM session state
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

# ----------------------------
# CHEMISTRY FUNCTIONS (Existing)
# ----------------------------

def get_compound_database():
    """Database of common compounds with SMILES"""
    return {
        # Alcohols (8)
        'benzyl alcohol': 'c1ccccc1CO',
        'ethanol': 'CCO',
        'methanol': 'CO',
        'cyclohexanol': 'OC1CCCCC1',
        'isopropanol': 'CC(C)O',
        '1-phenylethanol': 'CC(O)c1ccccc1',
        'tert-butanol': 'CC(C)(C)O',
        'ethylene glycol': 'OCCO',
        
        # Aldehydes (6)
        'benzaldehyde': 'c1ccccc1C=O',
        'acetaldehyde': 'CC=O',
        'formaldehyde': 'C=O',
        'propionaldehyde': 'CCC=O',
        'butyraldehyde': 'CCCC=O',
        'salicylaldehyde': 'Oc1ccccc1C=O',
        
        # Carboxylic acids (10)
        'benzoic acid': 'c1ccccc1C(=O)O',
        'acetic acid': 'CC(=O)O',
        'formic acid': 'OC=O',
        'propionic acid': 'CCC(=O)O',
        'butyric acid': 'CCCC(=O)O',
        'oxalic acid': 'OC(=O)C(=O)O',
        'malonic acid': 'OC(=O)CC(=O)O',
        'succinic acid': 'OC(=O)CCC(=O)O',
        'adipic acid': 'OC(=O)CCCCC(=O)O',
        'citric acid': 'OC(=O)CC(O)(CC(=O)O)C(=O)O',
        
        # Ketones (6)
        'acetophenone': 'CC(=O)c1ccccc1',
        'acetone': 'CC(=O)C',
        'cyclohexanone': 'O=C1CCCCC1',
        'butanone': 'CCC(=O)C',
        'benzophenone': 'O=C(c1ccccc1)c2ccccc2',
        'acetylacetone': 'CC(=O)CC(=O)C',
        
        # Aromatic compounds (12)
        'toluene': 'Cc1ccccc1',
        'benzene': 'c1ccccc1',
        'phenol': 'Oc1ccccc1',
        'aniline': 'Nc1ccccc1',
        'nitrobenzene': 'O=[N+]([O-])c1ccccc1',
        'bromobenzene': 'Brc1ccccc1',
        'chlorobenzene': 'Clc1ccccc1',
        'iodobenzene': 'Ic1ccccc1',
        'anisole': 'COc1ccccc1',
        'ethylbenzene': 'CCc1ccccc1',
        'styrene': 'C=Cc1ccccc1',
        'naphthalene': 'c1ccc2ccccc2c1',
        
        # Esters and derivatives (10)
        'methyl benzoate': 'COC(=O)c1ccccc1',
        'ethyl acetate': 'CCOC(=O)C',
        'benzyl acetate': 'COC(=O)c1ccccc1',
        'acetanilide': 'CC(=O)Nc1ccccc1',
        'acetyl chloride': 'CC(=O)Cl',
        'acetic anhydride': 'CC(=O)OC(=O)C',
        'methyl salicylate': 'COC(=O)c1ccccc1O',
        'ethyl benzoate': 'CCOC(=O)c1ccccc1',
        'butyl acetate': 'CCCCOC(=O)C',
        'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        
        # Amides and amines (10)
        'methylamine': 'CN',
        'dimethylamine': 'CNC',
        'trimethylamine': 'CN(C)C',
        'acetamide': 'CC(=O)N',
        'urea': 'NC(=O)N',
        'oxamide': 'NC(=O)C(=O)N',
        'malonamide': 'NC(=O)CC(=O)N',
        'benzylamine': 'NCc1ccccc1',
        'diethylamine': 'CCNCC',
        'ammonia': 'N',
        
        # Alkenes and alkanes (8)
        'ethylene': 'C=C',
        'acetylene': 'C#C',
        'cyclohexane': 'C1CCCCC1',
        'hexane': 'CCCCCC',
        'propene': 'C=CC',
        'butane': 'CCCC',
        'isobutane': 'CC(C)C',
        'cyclopentane': 'C1CCCC1',
        
        # Heterocycles (6)
        'pyridine': 'c1ccncc1',
        'furan': 'c1ccoc1',
        'thiophene': 'c1ccsc1',
        'pyrrole': 'c1cc[nH]c1',
        'imidazole': 'c1cnc[nH]1',
        'piperidine': 'C1CCNCC1'
    }

def get_reaction_pathways():
    """All available reaction pathways with multiple reactant support"""
    return {
        'oxidation': [
            {
                'name': 'Primary Alcohol to Carboxylic Acid',
                'reactants': ['c1ccccc1CO'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['c1ccccc1C(=O)O'],
                'description': 'Primary alcohol → Aldehyde → Carboxylic acid',
                'reagents': ['KMnO₄', 'K₂Cr₂O₇/H₂SO₄', 'Jones reagent'],
                'mechanism': 'Stepwise oxidation via chromate ester intermediate',
                'conditions': 'Heating, acidic or basic conditions'
            },
            {
                'name': 'Alkene to Diol (Dihydroxylation)',
                'reactants': ['C=C'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['C(O)C(O)'],
                'description': 'Alkene → Vicinal diol',
                'reagents': ['OsO₄/NMO', 'KMnO₄/OH⁻ cold'],
                'mechanism': 'Syn addition via osmate ester',
                'conditions': 'Cold, basic conditions'
            }
        ],
        'reduction': [
            {
                'name': 'Nitro to Amino Reduction',
                'reactants': ['O=[N+]([O-])c1ccccc1'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['Nc1ccccc1'],
                'description': 'Nitro group → Amino group',
                'reagents': ['Sn/HCl', 'Fe/HCl', 'H₂/Pd'],
                'mechanism': 'Stepwise reduction via nitroso and hydroxylamine intermediates',
                'conditions': 'Acidic conditions, heating'
            },
            {
                'name': 'Ketone to Alcohol Reduction',
                'reactants': ['CC(=O)c1ccccc1'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['CC(O)c1ccccc1'],
                'description': 'Ketone → Secondary alcohol',
                'reagents': ['NaBH₄', 'LiAlH₄'],
                'mechanism': 'Nucleophilic addition of hydride ion',
                'conditions': 'Anhydrous conditions for LiAlH₄'
            }
        ],
        'esterification': [
            {
                'name': 'Fischer Esterification',
                'reactants': ['c1ccccc1C(=O)O', 'CO'],
                'min_reactants': 2,
                'max_reactants': 2,
                'products': ['COC(=O)c1ccccc1'],
                'description': 'Carboxylic acid + Alcohol → Ester + Water',
                'reagents': ['H₂SO₄ (cat.)', 'HCl (cat.)'],
                'mechanism': 'Nucleophilic acyl substitution with acid catalysis',
                'conditions': 'Reflux, removal of water'
            }
        ],
        'hydrolysis': [
            {
                'name': 'Ester Hydrolysis (Basic)',
                'reactants': ['COC(=O)c1ccccc1'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['c1ccccc1C(=O)O', 'CO'],
                'description': 'Ester → Carboxylic acid salt + Alcohol',
                'reagents': ['NaOH/H₂O'],
                'mechanism': 'Nucleophilic acyl substitution (SN2)',
                'conditions': 'Reflux, aqueous conditions'
            },
            {
                'name': 'Amide Hydrolysis',
                'reactants': ['CC(=O)Nc1ccccc1'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['Nc1ccccc1', 'CC(=O)O'],
                'description': 'Amide → Amine + Carboxylic acid',
                'reagents': ['HCl/H₂O', 'NaOH/H₂O'],
                'mechanism': 'Nucleophilic acyl substitution under vigorous conditions',
                'conditions': 'Strong acid/base, heating'
            }
        ],
        'acetylation': [
            {
                'name': 'Amine Acetylation',
                'reactants': ['Nc1ccccc1', 'CC(=O)Cl'],
                'min_reactants': 2,
                'max_reactants': 2,
                'products': ['CC(=O)Nc1ccccc1'],
                'description': 'Amine + Acid chloride → Amide',
                'reagents': ['Acetyl chloride', 'Triethylamine'],
                'mechanism': 'Nucleophilic acyl substitution',
                'conditions': 'Anhydrous, base to scavenge HCl'
            }
        ],
        'halogenation': [
            {
                'name': 'Aromatic Bromination',
                'reactants': ['c1ccccc1'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['Brc1ccccc1'],
                'description': 'Benzene → Bromobenzene',
                'reagents': ['Br₂/FeBr₃'],
                'mechanism': 'Electrophilic aromatic substitution',
                'conditions': 'Room temperature, Lewis acid catalyst'
            }
        ],
        'nitration': [
            {
                'name': 'Aromatic Nitration',
                'reactants': ['c1ccccc1'],
                'min_reactants': 1,
                'max_reactants': 1,
                'products': ['O=[N+]([O-])c1ccccc1'],
                'description': 'Benzene → Nitrobenzene',
                'reagents': ['HNO₃/H₂SO₄'],
                'mechanism': 'Electrophilic aromatic substitution via nitronium ion',
                'conditions': 'Below 50°C, mixed acids'
            }
        ],
        'ammonolysis': [
            {
                'name': 'Oxalic Acid to Oxamide',
                'reactants': ['OC(=O)C(=O)O', 'N'],
                'min_reactants': 2,
                'max_reactants': 2,
                'products': ['NC(=O)C(=O)N'],
                'description': 'Oxalic acid + Concentrated ammonia → Oxamide',
                'reagents': ['NH₃ (conc.)', 'Heat'],
                'mechanism': 'Nucleophilic acyl substitution - Ammonia attacks carbonyl carbon, eliminating water to form oxamide',
                'conditions': 'Concentrated ammonia, heating, removal of water',
                'reaction_type': 'Condensation/Ammonolysis'
            }
        ]
    }

def validate_smiles(smiles):
    """Check if SMILES is valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def get_compound_name(smiles):
    """Get compound name from SMILES"""
    database = get_compound_database()
    for name, smi in database.items():
        if smi == smiles:
            return name.title()
    return "Unknown compound"

def draw_molecule(smiles, size=(200, 200)):
    """Draw a molecule from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Draw.MolToImage(mol, size=size)
    except:
        return None
    return None

def calculate_molecular_properties(smiles):
    """Calculate molecular properties"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} g/mol",
                'Formula': rdMolDescriptors.CalcMolFormula(mol),
                'Heavy Atoms': mol.GetNumHeavyAtoms(),
                'Atoms': mol.GetNumAtoms(),
                'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                'H-Bond Donors': Descriptors.NumHDonors(mol),
                'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
                'LogP': f"{Descriptors.MolLogP(mol):.2f}",
                'TPSA': f"{Descriptors.TPSA(mol):.2f} Å²"
            }
    except:
        return {}
    return None

def predict_product(reactant_smiles, reaction_type):
    """Predict product based on single reactant and reaction type"""
    prediction_rules = {
        'oxidation': {
            'c1ccccc1CO': 'c1ccccc1C=O',
            'CCO': 'CC=O',
            'CC=O': 'CC(=O)O',
            'c1ccccc1C=O': 'c1ccccc1C(=O)O',
        },
        'reduction': {
            'c1ccccc1C=O': 'c1ccccc1CO',
            'CC(=O)c1ccccc1': 'CC(O)c1ccccc1',
            'O=[N+]([O-])c1ccccc1': 'Nc1ccccc1',
            'CC(=O)O': 'CCO',
        },
        'esterification': {
            'c1ccccc1C(=O)O': 'COC(=O)c1ccccc1',
            'CC(=O)O': 'CCOC(=O)C',
        },
        'hydrolysis': {
            'COC(=O)c1ccccc1': 'c1ccccc1C(=O)O',
            'CCOC(=O)C': 'CC(=O)O',
            'CC(=O)Nc1ccccc1': 'Nc1ccccc1',
        },
        'acetylation': {
            'Nc1ccccc1': 'CC(=O)Nc1ccccc1',
        },
        'halogenation': {
            'c1ccccc1': 'Brc1ccccc1',
        },
        'nitration': {
            'c1ccccc1': 'O=[N+]([O-])c1ccccc1',
        },
        'ammonolysis': {
            'OC(=O)C(=O)O': 'NC(=O)C(=O)N',
        }
    }
    
    if reaction_type in prediction_rules:
        if reactant_smiles in prediction_rules[reaction_type]:
            return prediction_rules[reaction_type][reactant_smiles]
    return None

def predict_products_from_multiple(reactants, reaction_type):
    """Enhanced prediction for multiple reactants"""
    if reaction_type == "esterification" and len(reactants) >= 2:
        for i, r1 in enumerate(reactants):
            for j, r2 in enumerate(reactants):
                if i != j:
                    if 'C(=O)O' in r1 and 'CO' in r2:
                        if r1 == 'c1ccccc1C(=O)O' and r2 == 'CO':
                            return ['COC(=O)c1ccccc1']
                        elif r1 == 'CC(=O)O' and r2 == 'CCO':
                            return ['CCOC(=O)C']
    
    elif reaction_type == "ammonolysis" and len(reactants) >= 2:
        for i, r1 in enumerate(reactants):
            for j, r2 in enumerate(reactants):
                if i != j:
                    if (r1 == 'OC(=O)C(=O)O' or r1 == 'C(=O)(C(=O)O)O') and \
                       (r2 == 'N' or 'NH3' in r2.upper()):
                        return ['NC(=O)C(=O)N']
    
    predictions = []
    for reactant in reactants:
        predicted = predict_product(reactant, reaction_type)
        if predicted and predicted not in predictions:
            predictions.append(predicted)
    
    return predictions

def find_matching_pathways(reactants, reaction_type):
    """Find pathways that match the reactants and reaction type"""
    pathways = get_reaction_pathways()
    matching = []
    
    if reaction_type in pathways:
        for pathway in pathways[reaction_type]:
            if 'min_reactants' in pathway and 'max_reactants' in pathway:
                if pathway['min_reactants'] <= len(reactants) <= pathway['max_reactants']:
                    matching.append(pathway)
    
    return matching

# ----------------------------
# LLM FUNCTIONS
# ----------------------------

def call_llm(prompt, api_key=None, model=None, system_prompt=None):
    """Call LLM API"""
    if not api_key:
        return "⚠ Please provide an API key in the settings."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model or LLM_CONFIG["default_model"],
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(LLM_CONFIG["api_url"], headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error {response.status_code}: {response.text}"
    
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

def chemistry_expert_system_prompt():
    """System prompt for chemistry expert LLM"""
    return """You are an expert organic chemistry professor with 20+ years of experience.
    
    Your expertise includes:
    - Organic synthesis and retrosynthesis
    - Reaction mechanisms and arrow-pushing
    - Spectroscopy interpretation (NMR, IR, MS)
    - Physical organic chemistry
    - Named reactions and their applications
    
    Guidelines for your responses:
    1. Be accurate and detailed about chemical concepts
    2. Use proper chemical terminology and IUPAC names
    3. Explain mechanisms step-by-step when relevant
    4. Compare and contrast different synthetic routes
    5. Mention practical considerations (yield, conditions, safety)
    6. When discussing compounds, provide SMILES notation if possible
    7. Reference real literature examples when appropriate
    
    Always format your responses clearly with headings, bullet points, and chemical structures in text format."""

def analyze_with_llm(reaction_info, api_key, model):
    """Send reaction analysis to LLM for expert explanation"""
    
    prompt = f"""As a chemistry expert, analyze this reaction:

    REACTION INFORMATION:
    {json.dumps(reaction_info, indent=2)}
    
    Please provide:
    1. A detailed mechanism with electron-pushing arrows
    2. Explanation of regioselectivity/stereoselectivity if applicable
    3. Alternative synthetic routes to the same product
    4. Practical experimental considerations
    5. Potential side reactions or competing pathways
    6. Safety considerations for reagents
    
    Format your response with clear headings and bullet points."""
    
    return call_llm(prompt, api_key, model, chemistry_expert_system_prompt())

# ----------------------------
# MAIN APP
# ----------------------------

def main():
    st.set_page_config(
        page_title="Chemistry AI Assistant", 
        page_icon="🧪⚗", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session states
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Chemistry Solver"
    
    # Sidebar for app mode and LLM settings
    with st.sidebar:
        st.title("🧪⚗ Chemistry AI Assistant")
        
        # App mode selection
        st.session_state.app_mode = st.radio(
            "Select Mode:",
            ["Chemistry Solver", "AI Chemistry Assistant", "LLM Chat"],
            index=0
        )
        
        st.markdown("---")
        
        # LLM Settings (only for AI modes)
        if st.session_state.app_mode in ["AI Chemistry Assistant", "LLM Chat"]:
            st.subheader("🤖 LLM Configuration")
            
            LLM_CONFIG["api_key"] = st.text_input(
                "Groq API Key", 
                type="password",
                help="Get your API key from https://console.groq.com"
            )
            
            LLM_CONFIG["api_url"] = st.text_input(
                "API Endpoint",
                LLM_CONFIG["api_url"],
                help="Groq API endpoint (usually keep default)"
            )
            
            LLM_CONFIG["default_model"] = st.selectbox(
                "Model",
                ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
                index=0
            )
            
            st.markdown("---")
            
            # Clear chat history button
            if st.button("🗑 Clear Chat History"):
                st.session_state.llm_messages = []
                st.rerun()
        
        # Chemistry tools
        st.subheader("🔧 Chemistry Tools")
        
        # Quick SMILES validator
        with st.expander("SMILES Validator", expanded=False):
            test_smiles = st.text_input("Test SMILES:", "c1ccccc1C(=O)O")
            if test_smiles:
                if validate_smiles(test_smiles):
                    st.success("✅ Valid")
                    img = draw_molecule(test_smiles, (120, 120))
                    if img:
                        st.image(img, caption=get_compound_name(test_smiles))
                else:
                    st.error("❌ Invalid")
        
        # Quick compound lookup
        with st.expander("Compound Lookup", expanded=False):
            compound_query = st.selectbox(
                "Select compound:", 
                sorted(list(get_compound_database().keys())),
                index=0
            )
            if compound_query:
                smiles = get_compound_database()[compound_query]
                st.code(smiles)
                img = draw_molecule(smiles, (100, 100))
                if img:
                    st.image(img, caption=compound_query.title())
    
    # Main content area based on selected mode
    if st.session_state.app_mode == "Chemistry Solver":
        render_chemistry_solver()
    
    elif st.session_state.app_mode == "AI Chemistry Assistant":
        render_ai_assistant()
    
    elif st.session_state.app_mode == "LLM Chat":
        render_llm_chat()

# ----------------------------
# MODE 1: CHEMISTRY SOLVER
# ----------------------------

def render_chemistry_solver():
    st.title("🧪 Chemistry Problem Solver")
    st.markdown("Solve organic chemistry problems with 1-5 reactants")
    
    st.subheader("⚙ Reaction Configuration")
    
    col_config, col_preview = st.columns([1, 1])
    
    with col_config:
        num_reactants = st.slider(
            "*Number of reactants:*", 
            1, 5, 1,
            help="Select how many reactants (1-5)"
        )
        
        reaction_descriptions = {
            "oxidation": "Increase oxygen/decrease hydrogen",
            "reduction": "Decrease oxygen/increase hydrogen", 
            "esterification": "Acid + Alcohol → Ester",
            "hydrolysis": "Ester → Acid + Alcohol",
            "acetylation": "Add acetyl group",
            "halogenation": "Add halogen atom",
            "nitration": "Add nitro group",
            "ammonolysis": "Acid + Ammonia → Amide",
        }
        
        reaction_type = st.selectbox(
            "*Select reaction type:*",
            list(reaction_descriptions.keys()),
            format_func=lambda x: f"{x.title()} - {reaction_descriptions[x]}"
        )
    
    with col_preview:
        st.info(f"*Configuration:* {num_reactants} reactant(s), {reaction_type} reaction")
    
    st.markdown("---")
    
    # Reactant input
    st.markdown("### 📦 Reactant Input")
    reactants = []
    
    reactant_cols = st.columns(min(num_reactants, 3))
    default_examples = ['c1ccccc1CO', 'CC=O', 'CC(=O)O', 'Nc1ccccc1', 'Cc1ccccc1']
    
    for i in range(num_reactants):
        col_idx = i % 3
        with reactant_cols[col_idx]:
            default_smiles = default_examples[i] if i < len(default_examples) else ""
            r_smiles = st.text_input(
                f"*Reactant {i+1} SMILES*", 
                default_smiles,
                key=f"reactant_{i}"
            )
            
            if r_smiles:
                reactants.append(r_smiles)
                
                if r_smiles.strip():
                    if validate_smiles(r_smiles):
                        st.success(f"✅ {get_compound_name(r_smiles)}")
                        img = draw_molecule(r_smiles, (120, 120))
                        if img:
                            st.image(img, caption=f"R{i+1}")
                    else:
                        st.error("❌ Invalid SMILES")
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🔬 Analyze Reaction", type="primary", use_container_width=True):
        if not reactants:
            st.warning("Please enter at least one reactant!")
        else:
            with st.spinner("Analyzing reaction..."):
                
                # Display reactants
                st.subheader("🔬 Reaction Analysis")
                st.markdown("📦 Reactants:")
                
                num_r = len(reactants)
                cols_per_row = min(4, num_r)
                
                for row_start in range(0, num_r, cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for col_idx in range(cols_per_row):
                        r_idx = row_start + col_idx
                        if r_idx < num_r:
                            with cols[col_idx]:
                                r_smiles = reactants[r_idx]
                                if validate_smiles(r_smiles):
                                    st.write(f"*Reactant {r_idx+1}*")
                                    st.code(r_smiles)
                                    st.write(f"{get_compound_name(r_smiles)}")
                
                # Predict products
                st.markdown("🎯 Predicted Products:")
                predicted_products = predict_products_from_multiple(reactants, reaction_type)
                
                if predicted_products:
                    num_p = len(predicted_products)
                    p_cols_per_row = min(4, num_p)
                    
                    for row_start in range(0, num_p, p_cols_per_row):
                        cols = st.columns(p_cols_per_row)
                        
                        for col_idx in range(p_cols_per_row):
                            p_idx = row_start + col_idx
                            if p_idx < num_p:
                                with cols[col_idx]:
                                    p_smiles = predicted_products[p_idx]
                                    if validate_smiles(p_smiles):
                                        st.write(f"*Product {p_idx+1}*")
                                        st.code(p_smiles)
                                        st.write(f"{get_compound_name(p_smiles)}")
                else:
                    st.warning("No products predicted for this combination.")
                
                # Matching pathways
                st.markdown("📚 Matching Reaction Pathways:")
                matching_pathways = find_matching_pathways(reactants, reaction_type)
                
                if matching_pathways:
                    for pathway in matching_pathways:
                        with st.expander(f"🔬 {pathway['name']}"):
                            st.write(f"*Description:* {pathway['description']}")
                            
                            if 'reagents' in pathway:
                                st.write("*Reagents:*")
                                for reagent in pathway['reagents']:
                                    st.write(f"- {reagent}")
                            
                            if 'mechanism' in pathway:
                                st.write("*Mechanism:*")
                                st.info(pathway['mechanism'])
                else:
                    st.info(f"No predefined {reaction_type} pathways match your reactants.")

# ----------------------------
# MODE 2: AI CHEMISTRY ASSISTANT
# ----------------------------

def render_ai_assistant():
    st.title("🤖 AI Chemistry Assistant")
    st.markdown("Get expert explanations for chemistry problems using AI")
    
    if not LLM_CONFIG["api_key"]:
        st.warning("⚠ Please enter your Groq API key in the sidebar to use the AI Assistant.")
        return
    
    # Two-column layout: Chemistry input on left, LLM chat on right
    col_chem, col_llm = st.columns([1, 1])
    
    with col_chem:
        st.subheader("🧪 Chemistry Input")
        
        # Quick reaction input
        reaction_input = st.text_area(
            "Describe a reaction or ask a chemistry question:",
            height=150,
            placeholder="e.g., 'Explain the mechanism of oxalic acid ammonolysis to oxamide' or 'How do I synthesize aspirin from salicylic acid?'"
        )
        
        # Or use structured input
        with st.expander("Or use structured input"):
            num_reactants = st.slider("Reactants", 1, 3, 1)
            reactants = []
            for i in range(num_reactants):
                r = st.text_input(f"Reactant {i+1}", key=f"ai_reactant_{i}")
                if r:
                    reactants.append(r)
            
            reaction_type = st.selectbox("Reaction Type", 
                ["oxidation", "reduction", "esterification", "hydrolysis", 
                 "acetylation", "halogenation", "nitration", "ammonolysis"])
        
        if st.button("🔍 Analyze with AI", type="primary"):
            if reaction_input or reactants:
                with st.spinner("Consulting chemistry expert..."):
                    
                    # Prepare reaction info for LLM
                    reaction_info = {
                        "user_query": reaction_input,
                        "reactants": reactants if reactants else [],
                        "reaction_type": reaction_type if 'reaction_type' in locals() else None,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    # Call LLM
                    llm_response = analyze_with_llm(
                        reaction_info, 
                        LLM_CONFIG["api_key"], 
                        LLM_CONFIG["default_model"]
                    )
                    
                    # Store in session
                    st.session_state.llm_messages.append({
                        "role": "user",
                        "content": reaction_input or f"Analyze: {reactants} -> {reaction_type}"
                    })
                    st.session_state.llm_messages.append({
                        "role": "assistant",
                        "content": llm_response
                    })
    
    with col_llm:
        st.subheader("💬 AI Response")
        
        # Display chat history
        for msg in st.session_state.llm_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
        
        # Download chat history
        if st.session_state.llm_messages:
            chat_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in st.session_state.llm_messages
            ])
            
            st.download_button(
                label="📥 Download Conversation",
                data=chat_text,
                file_name="chemistry_ai_chat.txt",
                mime="text/plain"
            )

# ----------------------------
# MODE 3: LLM CHAT
# ----------------------------

def render_llm_chat():
    st.title("💬 General LLM Chat")
    
    if not LLM_CONFIG["api_key"]:
        st.warning("⚠ Please enter your Groq API key in the sidebar.")
        return
    
    # Chat interface
    user_input = st.text_area("Enter your message:", height=100)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("📤 Send Message", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("Thinking..."):
                    # Add user message
                    st.session_state.llm_messages.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get LLM response
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
    
    with col2:
        if st.button("🗑 Clear Chat", type="secondary", use_container_width=True):
            st.session_state.llm_messages = []
            st.rerun()
    
    st.markdown("---")
    st.subheader("💬 Conversation History")
    
    # Display messages
    for msg in st.session_state.llm_messages:
        if msg["role"] == "user":
            st.markdown(f"🧑 You:** {msg['content']}")
        else:
            st.markdown(f"🤖 Assistant:** {msg['content']}")
        
        st.markdown("---")
    
    # Download option
    if st.session_state.llm_messages:
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.llm_messages
        ])
        
        st.download_button(
            label="📥 Download Full Chat",
            data=chat_text,
            file_name="llm_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
st.markdown("[reference-wikipedia](https://en.wikipedia.org/wiki/List_of_organic_reactions)")
st.info("developed by subramanian ramajayam")
st.snow()
# ----------------------------
# RUN THE APP
# ----------------------------

if __name__ == "__main__":
    main()

