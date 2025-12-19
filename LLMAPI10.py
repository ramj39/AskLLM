#The syntax error is likely due to incorrect indentation. Let me provide the complete, properly formatted code. The issue is probably that the render_enhanced_sidebar() function is incorrectly placed. Here's the complete corrected code:
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
import pandas as pd
import requests
import json
import os
from typing import List, Dict
import hashlib
def get_api_key():
    # Prefer secrets; fallback to environment
    return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

api_key = get_api_key()

# Optional: export to environment for libraries that read it
os.environ["GROQ_API_KEY"] = api_key

# --- Sidebar: status only, never the key ---
st.sidebar.header("LLM configuration")
if api_key:
    st.sidebar.success("✅ Groq API Key active")
else:
    st.sidebar.error("❌ No API Key found")

# Optional override (does not show secrets, safe for public)
user_key = st.sidebar.text_input("Override API Key", type="password", value="")
if user_key:
    api_key = user_key

# --- Main UI: masked confirmation (optional) ---
show_masked = st.toggle("Show masked key", value=False)
if show_masked and api_key:
    st.write("Key loaded:", api_key[:5] + "..." + api_key[-3:])
elif not api_key:
    st.write("No key found.")
# ----------------------------
# ENHANCED CHEMISTRY FRAMEWORK
# ----------------------------

class ChemistryReactionEngine:
    """Enhanced reaction engine with pattern matching and rule-based predictions"""
    
    def __init__(self):
        self.reaction_rules = self._initialize_reaction_rules()
        self.functional_groups = self._initialize_functional_groups()
        self.smiles_fixes = self._initialize_smiles_fixes()
        self.compound_database = self._initialize_compound_database()
    
    def _initialize_smiles_fixes(self):
        """Fix common problematic SMILES"""
        return {
            'O=[N+]([O-])O': 'O=N(=O)O',  # Nitric acid
            'H2SO4': 'OS(=O)(=O)O',       # Sulfuric acid
            'HNO3': 'O=N(=O)O',           # Nitric acid
            'HCl': '[H+].[Cl-]',          # Hydrochloric acid
            'HBr': '[H+].[Br-]',          # Hydrobromic acid
            'HI': '[H+].[I-]',            # Hydroiodic acid
            'NaOH': '[Na+].[OH-]',        # Sodium hydroxide
            'KOH': '[K+].[OH-]',          # Potassium hydroxide
            'NH3': 'N',                   # Ammonia
            'NH4OH': '[NH4+].[OH-]',      # Ammonium hydroxide
            'NaCl': '[Na+].[Cl-]',        # Sodium chloride
            'KCl': '[K+].[Cl-]',          # Potassium chloride
            'CaCO3': '[Ca++].[O-]C(=O)[O-]',  # Calcium carbonate
            'Na2CO3': '[Na+].[Na+].[O-]C(=O)[O-]',  # Sodium carbonate
        }
    
    def _initialize_compound_database(self):
        """Comprehensive compound database"""
        return {
            # ALCOHOLS (Primary)
            'methanol': 'CO',
            'ethanol': 'CCO',
            'ethane':'CC',
            'propane':'CCC',
            'butane':'CCCC',
            'pentane':'CCCCC',
            'hexane':'CCCCCC',
            'heptane':'CCCCCCC',
            'octane':'CCCCCCCC',
            'nonane':'CCCCCCCCC',
            'decane':'CCCCCCCCCC',
            'ethylene':'C=C',
            'propylene':'CC#C',
            'butylene':'CCC#C',
            'terbutalene':'C1=CC(=CC(=C1)C(C)(C)C)N(C(C)C)CC(C)NCCO',
            'acenaphthylene':'C1=CC=CC2=C1C=CC3=C2C=CC=C3)',
            'methanol':'CO',
            'ethyl alcohol':'CCCO',
            '1-propanol': 'CCCO',
            '1-butanol': 'CCCCO',
            '1-pentanol': 'CCCCCO',
            '1-hexanol': 'CCCCCCO',
            '1-heptanol': 'CCCCCCCO',
            '1-octanol': 'CCCCCCCCO',
            '1-nonanol': 'CCCCCCCCCO',
            '1-decanol': 'CCCCCCCCCCO',
            'benzyl alcohol': 'c1ccccc1CO',
            'phenethyl alcohol': 'c1ccccc1CCO',
            'cinnamyl alcohol': 'c1ccccc1C=CCCO',
            
            # ALCOHOLS (Secondary)
            'isopropanol': 'CC(C)O',
            '2-butanol': 'CCC(C)O',
            '2-pentanol': 'CCCC(C)O',
            '2-hexanol': 'CCCCC(C)O',
            'cyclopentanol': 'OC1CCCC1',
            'cyclohexanol': 'OC1CCCCC1',
            '1-phenylethanol': 'CC(O)c1ccccc1',
            
            # ALCOHOLS (Tertiary)
            'tert-butanol': 'CC(C)(C)O',
            'tert-amyl alcohol': 'CCC(C)(C)O',
            '2-methyl-2-butanol': 'CCC(C)(C)O',
            
            # DIOLS AND TRIOLS
            'ethylene glycol': 'OCCO',
            '1,2-propanediol': 'CC(O)CO',
            '1,3-propanediol': 'OCCCO',
            'glycerol': 'OCC(O)CO',
            '1,2-butanediol': 'CCC(O)CO',
            '1,4-butanediol': 'OCCCCO',
            
            # PHENOLS
            'phenol': 'Oc1ccccc1',
            'acetamidophenol':'CC(=O)Nc1ccc(cc1)O',
            'catechol': 'Oc1ccccc1O',
            'resorcinol': 'Oc1cccc(O)c1',
            'hydroquinone': 'Oc1ccc(O)cc1',
            'p-cresol': 'Cc1ccc(O)cc1',
            'o-cresol': 'Cc1ccccc1O',
            'm-cresol': 'Cc1cccc(O)c1',
            'thymol': 'CC(C)c1ccc(O)cc1',
            
            # ALDEHYDES
            'formaldehyde': 'C=O',
            'acetaldehyde': 'CC=O',
            'propionaldehyde': 'CCC=O',
            'butyraldehyde': 'CCCC=O',
            'valeraldehyde': 'CCCCC=O',
            'benzaldehyde': 'c1ccccc1C=O',
            'salicylaldehyde': 'Oc1ccccc1C=O',
            'vanillin': 'COc1cc(C=O)ccc1O',
            'cinnamaldehyde': 'c1ccccc1C=CC=O',
            'furfural': 'c1ccoc1C=O',
            
            # KETONES
            'acetone': 'CC(=O)C',
            'aldol':'C(C(=O)C(=O)O)C(=O)C(=O)O',
            'acenaphthaquinone':'C1=CC=C2C(=C(C=C2C1=CC=CC=C3)O)C3=O',
            'acetoin':'CC(O)C(O)C',            
            'methyl ethyl ketone': 'CCC(=O)C',
            'diethyl ketone': 'CCC(=O)CC',
            'methyl isobutyl ketone': 'CC(C)CC(=O)C',
            'acetophenone': 'CC(=O)c1ccccc1',
            'benzophenone': 'O=C(c1ccccc1)c2ccccc2',
            'cyclohexanone': 'O=C1CCCCC1',
            'cyclopentanone': 'O=C1CCCC1',
            '2-butanone': 'CCC(=O)C',
            
            # CARBOXYLIC ACIDS
            'formic acid': 'OC=O',
            'acetic acid': 'CC(=O)O',
            'acetoacetic acid':'CC(=O)CC(=O)O',
            'alpha atropic acid':'C1=CC(=C(C(=C1)O)C(=O))C',
            'azeleic acid':'CC(C(=O)O)CCCCCC(=O)O',
            'beta atropic acid':'CC(=O)Nc1ccc(cc1)S(=O)(=O)N',
            'propionic acid': 'CCC(=O)O',
            'butyric acid': 'CCCC(=O)O',
            'valeric acid': 'CCCCC(=O)O',
            'caproic acid': 'CCCCCC(=O)O',
            'benzoic acid': 'c1ccccc1C(=O)O',
            'salicylic acid': 'Oc1ccccc1C(=O)O',
            'oxalic acid': 'OC(=O)C(=O)O',
            'malonic acid': 'OC(=O)CC(=O)O',
            'succinic acid': 'OC(=O)CCC(=O)O',
            'glutaric acid': 'OC(=O)CCCC(=O)O',
            'adipic acid': 'OC(=O)CCCCC(=O)O',
            'citric acid': 'OC(=O)CC(O)(CC(=O)O)C(=O)O',
            'lactic acid': 'CC(O)C(=O)O',
            'tartaric acid': 'OC(C(O)C(=O)O)C(=O)O',
            
            # ESTERS
            'methyl acetate': 'COC(=O)C',
            'ethyl acetate': 'CCOC(=O)C',
            'propyl acetate': 'CCCOC(=O)C',
            'butyl acetate': 'CCCCOC(=O)C',
            'methyl benzoate': 'COC(=O)c1ccccc1',
            'ethyl benzoate': 'CCOC(=O)c1ccccc1',
            'benzyl acetate': 'COC(=O)c1ccccc1',
            'methyl salicylate': 'COC(=O)c1ccccc1O',
            'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
            'ethyl acetoacetate': 'CC(=O)CC(=O)OCC',
            'ethyl suberate':'CC(C(=O)OCC)C(=O)OCC',
            
            # AMINES (Primary)
            'methylamine': 'CN',
            'ethylamine': 'CCN',
            'propylamine': 'CCCN',
            'butylamine': 'CCCCN',
            'benzylamine': 'Nc1ccccc1',
            'aniline': 'Nc1ccccc1',
            '2-aminophenol': 'Nc1ccccc1O',
            '3-aminophenol': 'Nc1cccc(O)c1',
            '4-aminophenol': 'Nc1ccc(O)cc1',
            
            # AMINES (Secondary)
            'dimethylamine': 'CNC',
            'diethylamine': 'CCNCC',
            'dipropylamine': 'CCCNCCC',
            'methylaniline': 'CNc1ccccc1',
            'ethylaniline': 'CCNc1ccccc1',
            'azobenzene':'C1=CC=C(C=C1)N=Nc2ccccc2',
            
            # AMINES (Tertiary)
            'trimethylamine': 'CN(C)C',
            'triethylamine': 'CCN(CC)CC',
            'tripropylamine': 'CCCN(CCC)CCC',
            'N,N-dimethylaniline': 'CN(C)c1ccccc1',
            
            # AMIDES
            'formamide': 'NC=O',
            'acetamide': 'CC(=O)N',
            'acetoacetamide':' CC(=O)NCC(=O)C',
            'propionamide': 'CCC(=O)N',
            'benzamide': 'c1ccccc1C(=O)N',
            'acetanilide': 'CC(=O)Nc1ccccc1',
            'urea': 'NC(=O)N',
            'oxamide': 'NC(=O)C(=O)N',
            'malonamide': 'NC(=O)CC(=O)N',
            'succinamide': 'NC(=O)CCC(=O)N',
            
            # NITRILES
            'acetonitrile': 'CC#N',
            'propionitrile': 'CCC#N',
            'butyronitrile': 'CCCC#N',
            'benzonitrile': 'c1ccccc1C#N',
            'phenylacetonitrile': 'c1ccccc1CC#N',
            
            # NITRO COMPOUNDS
            'nitromethane': 'C[N+](=O)[O-]',
            'ethyl nitrate':'CCO[N+](=O)[O-]',
            'nitroethane': 'CC[N+](=O)[O-]',
            '1-nitropropane': 'CCC[N+](=O)[O-]',
            'nitrobenzene': 'O=[N+]([O-])c1ccccc1',
            '2-nitrotoluene': 'Cc1ccccc1[N+](=O)[O-]',
            '3-nitrotoluene': 'Cc1cccc([N+](=O)[O-])c1',
            '4-nitrotoluene': 'Cc1ccc([N+](=O)[O-])cc1',
            '2,4-dinitrotoluene': 'Cc1cc([N+](=O)[O-])ccc1[N+](=O)[O-]',
            '2,4,6-trinitrotoluene': 'Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
            
            # HALIDES
            'methyl chloride': 'CCl',
            'ethyl chloride': 'CCCl',
            'propyl chloride': 'CCCCl',
            'butyl chloride': 'CCCCCl',
            'benzyl chloride': 'ClCc1ccccc1',
            'chlorobenzene': 'Clc1ccccc1',
            'bromobenzene': 'Brc1ccccc1',
            'iodobenzene': 'Ic1ccccc1',
            'fluorobenzene': 'Fc1ccccc1',
            '1,2-dichloroethane': 'ClCCCl',
            'chloroform': 'ClC(Cl)Cl',
            'carbon tetrachloride': 'ClC(Cl)(Cl)Cl',
            
            # ETHERES
            'diethyl ether': 'CCOCC',
            'methyl tert-butyl ether': 'COC(C)(C)C',
            'anisole': 'COc1ccccc1',
            'phenetole': 'CCOc1ccccc1',
            'diphenyl ether': 'O(c1ccccc1)c2ccccc2',
            'tetrahydrofuran': 'C1CCOC1',
            '1,4-dioxane': 'C1COCCO1',
            
            # ALKANES
            'methane': 'C',
            'ethane': 'CC',
            'propane': 'CCC',
            'butane': 'CCCC',
            'pentane': 'CCCCC',
            'hexane': 'CCCCCC',
            'heptane': 'CCCCCCC',
            'octane': 'CCCCCCCC',
            'nonane': 'CCCCCCCCC',
            'decane': 'CCCCCCCCCC',
            'cyclohexane': 'C1CCCCC1',
            'cyclopentane': 'C1CCCC1',
            'methylcyclohexane': 'CC1CCCCC1',
            
            # ALKENES
            'ethylene': 'C=C',
            'propene': 'C=CC',
            '1-butene': 'C=CCC',
            '2-butene': 'CC=CC',
            'isobutylene': 'C=C(C)C',
            '1-pentene': 'C=CCCC',
            '1-hexene': 'C=CCCCC',
            'styrene': 'C=Cc1ccccc1',
            'cyclopentene': 'C1=CCCC1',
            'cyclohexene': 'C1=CCCCC1',
            
            # ALKYNES
            'acetylene': 'C#C',
            'propylene': 'CC#C',
            '1-butyne': 'C#CCC',
            '2-butyne': 'CC#CC',
            '1-pentyne': 'C#CCCC',
            'phenylacetylene': 'c1ccccc1C#C',
            
            # AROMATICS
            'benzene': 'c1ccccc1',
            'toluene': 'Cc1ccccc1',
            'acetotoluene':'CC(C)(C)C(=O)C(C)C',
            'ethylbenzene': 'CCc1ccccc1',
            'xylene (o-)': 'Cc1ccccc1C',
            'xylene (m-)': 'Cc1cccc(C)c1',
            'xylene (p-)': 'Cc1ccc(C)cc1',
            'mesitylene': 'Cc1cc(C)cc(C)c1',
            'cumene': 'CC(C)c1ccccc1',
            'naphthalene': 'c1ccc2ccccc2c1',
            'anthracene': 'c1ccc2cc3ccccc3cc2c1',
            'phenanthrene': 'c1ccc2c(c1)ccc3ccccc32',
            
            # HETEROCYCLES
            'pyridine': 'c1ccncc1',
            'pyrrole': 'c1cc[nH]c1',
            'furan': 'c1ccoc1',
            'thiophene': 'c1ccsc1',
            'imidazole': 'c1cnc[nH]1',
            'pyrazole': 'c1ccnn1',
            'oxazole': 'c1cocn1',
            'thiazole': 'c1cscn1',
            'pyrimidine': 'c1cncnc1',
            'purine': 'c1c2c(nc[nH]2)ncn1',
            'indole': 'c1ccc2c(c1)[nH]c2',
            'quinoline': 'c1ccc2c(c1)nccc2',
            'isoquinoline': 'c1ccc2cnccc2c1',
            'piperidine': 'C1CCNCC1',
            'morpholine': 'C1COCCN1',
            'piperazine': 'C1CNCCN1',
            
            # INORGANIC COMPOUNDS
            'water': 'O',
            'ammonia': 'N',
            'hydrogen peroxide': 'OO',
            'nitric acid': 'O=N(=O)O',
            'sulfuric acid': 'OS(=O)(=O)O',
            'hydrochloric acid': '[H+].[Cl-]',
            'hydrobromic acid': '[H+].[Br-]',
            'hydroiodic acid': '[H+].[I-]',
            'phosphoric acid': 'OP(=O)(O)O',
            'boric acid': 'OB(O)O',
            'carbonic acid': 'OC(=O)O',
            'sodium hydroxide': '[Na+].[OH-]',
            'potassium hydroxide': '[K+].[OH-]',
            'calcium hydroxide': '[Ca++].[OH-].[OH-]',
            'sodium chloride': '[Na+].[Cl-]',
            'potassium chloride': '[K+].[Cl-]',
            'calcium chloride': '[Cl-].[Ca++].[Cl-]',
            'sodium carbonate': '[Na+].[Na+].[O-]C(=O)[O-]',
            'calcium carbonate': '[Ca++].[O-]C(=O)[O-]',
            'sodium bicarbonate': '[Na+].OC(=O)[O-]',
            
            # REAGENTS AND CATALYSTS
            'bromine': 'BrBr',
            'chlorine': 'ClCl',
            'iodine': 'II',
            'potassium permanganate': '[K+].[O-][Mn](=O)(=O)=O',
            'potassium dichromate': '[K+].[K+].[O-][Cr](=O)(=O)O[Cr](=O)(=O)[O-]',
            'sodium borohydride': '[Na+].[BH4-]',
            'lithium aluminum hydride': '[Li+].[AlH4-]',
            'sodium cyanoborohydride': '[Na+].[BH3-]C#N',
            'borane': '[BH3]',
            'diborane': 'B.B',
            'hydrogen gas': '[H][H]',
            'oxygen gas': 'O=O',
            'nitrogen gas': 'N#N',
            'carbon monoxide': '[C-]#[O+]',
            'carbon dioxide': 'O=C=O',
            'sulfur dioxide': 'O=S=O',
            'nitrogen dioxide': '[N+](=O)[O-]',
            'sulfur trioxide': 'O=S(=O)=O',
            'phosphorus pentachloride': 'ClP(Cl)(Cl)(Cl)Cl',
            'phosphorus oxychloride': 'O=P(Cl)(Cl)Cl',
            'thionyl chloride': 'O=S(Cl)Cl',
            'sulfonyl chloride': 'O=S(=O)(Cl)Cl',
            'acetyl chloride': 'CC(=O)Cl',
            'benzoyl chloride': 'c1ccccc1C(=O)Cl',
            'acetic anhydride': 'CC(=O)OC(=O)C',
            'benzoyl peroxide': 'O=C(Oc1ccccc1C(=O)O)c2ccccc2',
            'dicyclohexylcarbodiimide': 'C1CCCCC1N=C=NC2CCCCC2',
            'N-bromosuccinimide': 'O=C1CCC(=O)N1Br',
            'N-chlorosuccinimide': 'O=C1CCC(=O)N1Cl',
            'iodine monochloride': 'ICl',
            'boron trifluoride': 'FB(F)F',
            'aluminum chloride': '[Al+3].[Cl-].[Cl-].[Cl-]',
            'iron(III) chloride': '[Fe+3].[Cl-].[Cl-].[Cl-]',
            'tin(II) chloride': '[Cl-].[Sn+2].[Cl-]',
            'tin(IV) chloride': '[Cl-].[Sn+4].[Cl-].[Cl-].[Cl-]',
            'zinc chloride': '[Zn+2].[Cl-].[Cl-]',
            'copper(II) chloride': '[Cu+2].[Cl-].[Cl-]',
            'silver nitrate': '[Ag+].[O-][N+](=O)[O-]',
            'mercury(II) chloride': '[Cl-].[Hg+2].[Cl-]',
            'palladium on carbon': '[Pd].[C]',
            'platinum on carbon': '[Pt].[C]',
            'nickel catalyst': '[Ni]',
            'osmium tetroxide': 'O=[Os](=O)(=O)=O',
            'ruthenium tetroxide': 'O=[Ru](=O)(=O)=O',
            'potassium tert-butoxide': '[K+].CC(C)(C)[O-]',
            'sodium methoxide': '[Na+].CO',
            'sodium ethoxide': '[Na+].CCO',
            'lithium diisopropylamide': '[Li+].CC(C)N(C(C)C)[C-](C(C)C)C(C)C',
            'n-butyllithium': '[Li+].CCCC',
            'methylmagnesium bromide': '[Mg+]Br.C',
            'ethylmagnesium bromide': '[Mg+]Br.CC',
            'phenylmagnesium bromide': '[Mg+]Br.c1ccccc1',
            
            # AMINO ACIDS
            'glycine': 'NCC(=O)O',
            'alanine': 'CC(N)C(=O)O',
            'valine': 'CC(C)C(N)C(=O)O',
            'leucine': 'CC(C)CC(N)C(=O)O',
            'isoleucine': 'CCC(C)C(N)C(=O)O',
            'phenylalanine': 'c1ccccc1CC(N)C(=O)O',
            'tyrosine': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
            'tryptophan': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
            'serine': 'C(C(C(=O)O)N)O',
            'threonine': 'CC(C(C(=O)O)N)O',
            'cysteine': 'C(C(C(=O)O)N)S',
            'methionine': 'CSCCC(C(=O)O)N',
            'aspartic acid': 'C(C(C(=O)O)N)C(=O)O',
            'glutamic acid': 'C(CC(=O)O)C(C(=O)O)N',
            'asparagine': 'C(C(C(=O)O)N)C(=O)N',
            'glutamine': 'C(CC(=O)N)C(C(=O)O)N',
            'lysine': 'C(CCN)CC(C(=O)O)N',
            'arginine': 'C(CC(C(=O)O)N)CNC(=N)N',
            'histidine': 'c1c(nc[nH]1)CC(C(=O)O)N',
            'proline': 'C1CC(NC1)C(=O)O',
            
            # SUGARS
            'glucose': 'OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O',
            'fructose': 'OC[C@@H](O)[C@H](O)[C@H](O)C(=O)CO',
            'galactose': 'OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O',
            'mannose': 'OC[C@H]1O[C@H](O)[C@H](O)[C@H](O)[C@@H]1O',
            'ribose': 'OC[C@H]1O[C@H](O)[C@H](O)[C@H]1O',
            'deoxyribose': 'OC[C@H]1O[C@H](O)[C@H](O)[C@H]1C',
            'sucrose': 'OC[C@H]1O[C@H](OC[C@H]2O[C@H](O)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O',
            'lactose': 'OC[C@H]1O[C@H](OC[C@H]2O[C@H](O)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O',
            'maltose': 'OC[C@H]1O[C@H](OC[C@H]2O[C@H](O)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O',
            
            # COMMON DRUGS
            'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
            'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'paracetamol': 'CC(=O)Nc1ccc(O)cc1',
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'nicotine': 'CN1CCC[C@H]1c2cccnc2',
            'morphine': 'CN1CC[C@]23c4c5ccc(c4O[C@H]2[C@H](C=C[C@H]3[C@H]1C5)O)O',
            'codeine': 'CN1CC[C@]23c4c5ccc(c4O[C@H]2[C@H](C=C[C@H]3[C@H]1C5)OC)O',
            'cocaine': 'COC(=O)C1C2CCC(CC1OC(=O)c3ccccc3)N2C',
            'tetracycline': 'CN(C)C1C(=O)C2C(C3C(C(C2(O)C)O)(C(C(C3O)(C(=O)C(=C(C1(C)O)O)N(C)C)O)O)O)O',
            'penicillin G': 'CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C',
            'amoxicillin': 'CC1(C(N2C(S1)C(C2=O)NC(=O)C(Cc3ccc(O)cc3)N)C(=O)O)C',
        }
    
    def _initialize_functional_groups(self):
        """Define functional group patterns for smart matching"""
        return {
            'alcohol_primary': '[CX4][OH]',
            'alcohol_secondary': '[CX4]([CX4])[OH]',
            'alcohol_tertiary': '[CX4]([CX4])([CX4])[OH]',
            'phenol': 'Oc1ccccc1',
            'aldehyde': '[CX3H1](=O)[#6]',
            'ketone': '[#6][CX3](=O)[#6]',
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'ester': '[#6][CX3](=O)[OX2][#6]',
            'amine_primary': '[NX3;H2;!$(NC=O)]',
            'amine_secondary': '[NX3;H1;!$(NC=O)]',
            'amine_tertiary': '[NX3;H0;!$(NC=O)]',
            'amide': '[NX3][CX3](=O)',
            'nitro': '[N+](=O)[O-]',
            'nitrile': '[NX1]#[CX2]',
            'alkene': '[CX3]=[CX3]',
            'alkyne': '[CX2]#[CX2]',
            'aryl': 'c1ccccc1',
            'halide': '[F,Cl,Br,I]',
            'aniline': 'Nc1ccccc1',
            'benzaldehyde': 'c1ccccc1C=O',
            'benzoic_acid': 'c1ccccc1C(=O)O',
            'aryl_nitro': 'O=[N+]([O-])c1ccccc1',
            'aryl_halide': '[F,Cl,Br,I]c1ccccc1',
            'ether': '[OD2]([#6])[#6]',
            'thiol': '[SH]',
            'sulfide': '[SD2]([#6])[#6]',
        }
    
    def _initialize_reaction_rules(self):
        """Comprehensive reaction rules with mechanism patterns"""
        return {
            # NITRATION RULES
            'nitration': [
                {
                    'name': 'Aromatic Nitration (General)',
                    'reactant_patterns': ['c1ccccc1', 'O=N(=O)O'],
                    'min_reactants': 2,
                    'max_reactants': 3,
                    'catalyst': ['OS(=O)(=O)O'],  # H2SO4
                    'products': ['O=[N+]([O-])c1ccccc1'],
                    'mechanism': 'Electrophilic aromatic substitution via NO₂⁺',
                    'conditions': '50-60°C, mixed acids (HNO₃/H₂SO₄)',
                    'regioselectivity': 'Depends on substituents',
                    'yield': 'High',
                    'reaction_class': 'EAS',
                },
                {
                    'name': 'Phenol Nitration',
                    'reactant_patterns': ['Oc1ccccc1', 'O=N(=O)O'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'products': ['Oc1ccc(O)cc1[N+](=O)[O-]', 'O=[N+]([O-])c1ccc(O)cc1'],
                    'mechanism': 'Electrophilic aromatic substitution, ortho/para directing -OH',
                    'conditions': 'Dilute HNO₃, room temperature',
                    'regioselectivity': 'ortho/para mixture',
                    'yield': 'Medium',
                    'reaction_class': 'EAS',
                    'notes': 'Gives ortho and para isomers',
                },
                {
                    'name': 'Toluene Nitration',
                    'reactant_patterns': ['Cc1ccccc1', 'O=N(=O)O'],
                    'min_reactants': 2,
                    'max_reactants': 3,
                    'products': ['Cc1ccc([N+](=O)[O-])cc1', 'Cc1cc([N+](=O)[O-])ccc1'],
                    'mechanism': 'Electrophilic aromatic substitution, ortho/para directing -CH₃',
                    'conditions': 'Mixed acids, 30-50°C',
                    'regioselectivity': 'ortho/para mixture (mainly para)',
                    'yield': 'High',
                    'reaction_class': 'EAS',
                },
            ],
            
            # OXIDATION RULES
            'oxidation': [
                {
                    'name': 'Primary Alcohol to Aldehyde',
                    'reactant_patterns': ['[CX4][OH]'],
                    'min_reactants': 1,
                    'max_reactants': 1,
                    'reagents': ['PCC', 'DMP', 'Swern'],
                    'products': ['[CX3H1](=O)[#6]'],
                    'mechanism': 'Chromium-based oxidation',
                    'conditions': 'Anhydrous, room temp',
                    'reaction_class': 'Oxidation',
                },
                {
                    'name': 'Primary Alcohol to Carboxylic Acid',
                    'reactant_patterns': ['[CX4][OH]'],
                    'min_reactants': 1,
                    'max_reactants': 1,
                    'reagents': ['KMnO₄', 'K₂Cr₂O₇/H₂SO₄', 'Jones reagent'],
                    'products': ['[CX3](=O)[OX2H1]'],
                    'mechanism': 'Stepwise oxidation via aldehyde',
                    'conditions': 'Aqueous, heating',
                    'reaction_class': 'Oxidation',
                },
                {
                    'name': 'Secondary Alcohol to Ketone',
                    'reactant_patterns': ['[#6][CX4]([#6])[OH]'],
                    'min_reactants': 1,
                    'max_reactants': 1,
                    'reagents': ['K₂Cr₂O₇/H₂SO₄', 'PCC', 'Dess-Martin'],
                    'products': ['[#6][CX3](=O)[#6]'],
                    'mechanism': 'Chromium-based oxidation',
                    'conditions': 'Room temperature',
                    'reaction_class': 'Oxidation',
                },
            ],
            
            # REDUCTION RULES
            'reduction': [
                {
                    'name': 'Nitro to Amino Reduction',
                    'reactant_patterns': ['[N+](=O)[O-]'],
                    'min_reactants': 1,
                    'max_reactants': 1,
                    'reagents': ['Sn/HCl', 'Fe/HCl', 'H₂/Pd', 'Na₂S'],
                    'products': ['[NX3;H2]'],
                    'mechanism': 'Stepwise reduction via nitroso and hydroxylamine',
                    'conditions': 'Acidic conditions, heating',
                    'reaction_class': 'Reduction',
                },
                {
                    'name': 'Ketone to Alcohol Reduction',
                    'reactant_patterns': ['[#6][CX3](=O)[#6]'],
                    'min_reactants': 1,
                    'max_reactants': 1,
                    'reagents': ['NaBH₄', 'LiAlH₄'],
                    'products': ['[#6][CX4]([OH])[#6]'],
                    'mechanism': 'Nucleophilic addition of hydride',
                    'conditions': 'Anhydrous for LiAlH₄',
                    'reaction_class': 'Reduction',
                },
            ],
            
            # ESTERIFICATION RULES
            'esterification': [
                {
                    'name': 'Fischer Esterification',
                    'reactant_patterns': ['[CX3](=O)[OX2H1]', '[CX4][OH]'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'catalyst': ['[H+]', 'OS(=O)(=O)O'],
                    'products': ['[#6][CX3](=O)[OX2][#6]'],
                    'mechanism': 'Nucleophilic acyl substitution',
                    'conditions': 'Acid catalyst, heating',
                    'reaction_class': 'Condensation',
                    'reversible': True,
                },
            ],
            
            # HYDROLYSIS RULES
            'hydrolysis': [
                {
                    'name': 'Ester Hydrolysis (Acidic)',
                    'reactant_patterns': ['[#6][CX3](=O)[OX2][#6]'],
                    'min_reactants': 1,
                    'max_reactants': 1,
                    'reagents': ['[H+]', 'OS(=O)(=O)O'],
                    'products': ['[CX3](=O)[OX2H1]', '[CX4][OH]'],
                    'mechanism': 'Nucleophilic acyl substitution',
                    'conditions': 'Acid, heating',
                    'reaction_class': 'Hydrolysis',
                },
            ],
            
            # ACYLATION RULES
            'acetylation': [
                {
                    'name': 'Amine Acetylation',
                    'reactant_patterns': ['[NX3;H2,H1]', '[CX3](=O)[Cl]'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'base': ['pyridine', '[NX3]'],
                    'products': ['[NX3][CX3](=O)'],
                    'mechanism': 'Nucleophilic acyl substitution',
                    'conditions': 'Base, room temperature',
                    'reaction_class': 'Acylation',
                },
            ],
            
            # HALOGENATION RULES
            'halogenation': [
                {
                    'name': 'Aromatic Halogenation',
                    'reactant_patterns': ['c1ccccc1', '[Br]Br', '[Cl]Cl'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'catalyst': ['Fe', 'FeCl₃', 'AlCl₃'],
                    'products': ['[Br,Cl]c1ccccc1'],
                    'mechanism': 'Electrophilic aromatic substitution',
                    'conditions': 'Room temperature, catalyst',
                    'reaction_class': 'EAS',
                },
            ],
            
            # OTHER IMPORTANT REACTIONS
            'ammonolysis': [
                {
                    'name': 'Acid Chloride + Ammonia',
                    'reactant_patterns': ['[CX3](=O)[Cl]', 'N'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'products': ['[NX3][CX3](=O)'],
                    'mechanism': 'Nucleophilic acyl substitution',
                    'conditions': 'Room temperature',
                    'reaction_class': 'Acylation',
                },
            ],
            
            'friedel_crafts_alkylation': [
                {
                    'name': 'Friedel-Crafts Alkylation',
                    'reactant_patterns': ['c1ccccc1', '[CX4][Cl,Br]'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'catalyst': ['AlCl₃', 'FeCl₃'],
                    'products': ['c1ccccc1[CX4]'],
                    'mechanism': 'Electrophilic aromatic substitution via carbocation',
                    'conditions': 'Anhydrous, Lewis acid catalyst',
                    'reaction_class': 'EAS',
                },
            ],
            
            'sulfonation': [
                {
                    'name': 'Aromatic Sulfonation',
                    'reactant_patterns': ['c1ccccc1', 'OS(=O)(=O)O'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'products': ['c1ccc(S(=O)(=O)O)cc1'],
                    'mechanism': 'Electrophilic aromatic substitution',
                    'conditions': 'Fuming H₂SO₄, heating',
                    'reaction_class': 'EAS',
                    'reversible': True,
                },
            ],
            
            'grignard': [
                {
                    'name': 'Grignard with Aldehyde/Ketone',
                    'reactant_patterns': ['[Mg][Br,Cl]', '[CX3](=O)[#6]'],
                    'min_reactants': 2,
                    'max_reactants': 2,
                    'products': ['[#6][CX4]([OH])[#6]'],
                    'mechanism': 'Nucleophilic addition',
                    'conditions': 'Anhydrous ether, then acidic workup',
                    'reaction_class': 'Addition',
                },
            ],
        }
    
    # ADD THE MISSING METHOD
    def get_compound_groups(self):
        """Get compounds organized by functional groups"""
        # Initialize all categories
        groups = {
            'Alcohols': {},
            'Aldehydes': {},
            'Ketones': {},
            'Carboxylic Acids': {},
            'Esters': {},
            'Amines': {},
            'Amides': {},
            'Nitro Compounds': {},
            'Halides': {},
            'Ethers': {},
            'Alkanes': {},
            'Alkenes': {},
            'Alkynes': {},
            'Aromatics': {},
            'Heterocycles': {},
            'Inorganic Compounds': {},
            'Reagents and Catalysts': {},
            'Amino Acids': {},
            'Sugars': {},
            'Common Drugs': {},
            'Other': {},
        }
        
        for name, smiles in self.compound_database.items():
            try:
                fixed_smiles = self.fix_smiles(smiles)
                
                # SIMPLIFIED CATEGORIZATION
                if '[' in fixed_smiles and ']' in fixed_smiles:
                    groups['Inorganic Compounds'][name] = smiles
                elif 'OH' in fixed_smiles:
                    groups['Alcohols'][name] = smiles
                elif 'C=O' in fixed_smiles:
                    if 'C(=O)O' in fixed_smiles:
                        groups['Carboxylic Acids'][name] = smiles
                    else:
                        groups['Aldehydes'][name] = smiles
                elif 'C(=O)C' in fixed_smiles:
                    groups['Ketones'][name] = smiles
                elif 'OC(=O)' in fixed_smiles:
                    groups['Esters'][name] = smiles
                elif 'N' in fixed_smiles:
                    if 'C(=O)N' in fixed_smiles:
                        groups['Amides'][name] = smiles
                    else:
                        groups['Amines'][name] = smiles
                elif 'c1ccccc1' in fixed_smiles:
                    groups['Aromatics'][name] = smiles
                elif any(hal in fixed_smiles for hal in ['Cl', 'Br', 'I', 'F']):
                    groups['Halides'][name] = smiles
                elif any(keyword in name.lower() for keyword in ['amino', 'glycine', 'alanine', 'valine']):
                    groups['Amino Acids'][name] = smiles
                elif any(keyword in name.lower() for keyword in ['glucose', 'fructose', 'sucrose', 'lactose']):
                    groups['Sugars'][name] = smiles
                elif any(keyword in name.lower() for keyword in ['aspirin', 'ibuprofen', 'paracetamol', 'caffeine']):
                    groups['Common Drugs'][name] = smiles
                elif any(keyword in name.lower() for keyword in ['reagent', 'catalyst', 'borohydride', 'permanganate']):
                    groups['Reagents and Catalysts'][name] = smiles
                else:
                    groups['Other'][name] = smiles
                    
            except:
                groups['Other'][name] = smiles
        
        # Remove empty categories
        return {k: v for k, v in groups.items() if v}
    
    def fix_smiles(self, smiles: str) -> str:
        """Fix common SMILES issues"""
        return self.smiles_fixes.get(smiles, smiles)
    
    def get_compound_database(self):
        """Get the comprehensive compound database"""
        return self.compound_database
    
    def validate_and_fix(self, smiles_list: List[str]) -> List[str]:
        """Validate and fix a list of SMILES"""
        fixed = []
        for smi in smiles_list:
            fixed_smi = self.fix_smiles(smi)
            if Chem.MolFromSmiles(fixed_smi):
                fixed.append(fixed_smi)
            else:
                fixed_smi = smi.replace('O=[N+]([O-])O', 'O=N(=O)O')
                if Chem.MolFromSmiles(fixed_smi):
                    fixed.append(fixed_smi)
                else:
                    fixed.append(smi)
        return fixed
    
    def has_functional_group(self, smiles: str, group_name: str) -> bool:
        """Check if molecule has specific functional group"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        
        if group_name in self.functional_groups:
            pattern = Chem.MolFromSmarts(self.functional_groups[group_name])
            return mol.HasSubstructMatch(pattern)
        return False
    
    def get_functional_groups(self, smiles: str) -> List[str]:
        """Get all functional groups present in molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        groups = []
        for name, pattern in self.functional_groups.items():
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                groups.append(name)
        return groups
    
    def predict_reaction(self, reactants: List[str], reaction_type: str) -> Dict:
        """Predict reaction products with detailed information"""
        # Fix SMILES first
        reactants = self.validate_and_fix(reactants)
        
        # Initialize result structure
        result = {
            'success': False,
            'reaction_type': reaction_type,
            'reactants': reactants,
            'products': [],
            'matched_rules': [],
            'mechanism': '',
            'conditions': '',
            'notes': [],
            'warnings': [],
        }
        
        # Check if reaction type exists
        if reaction_type not in self.reaction_rules:
            result['warnings'].append(f"Unknown reaction type: {reaction_type}")
            return result
        
        # Get rules for this reaction type
        rules = self.reaction_rules[reaction_type]
        
        # Try to match with each rule
        for rule in rules:
            if rule['min_reactants'] <= len(reactants) <= rule['max_reactants']:
                matched = True
                for pattern in rule.get('reactant_patterns', []):
                    pattern_found = False
                    for reactant in reactants:
                        if pattern in reactant:
                            pattern_found = True
                            break
                    if not pattern_found:
                        matched = False
                        break
                
                if matched:
                    result['matched_rules'].append(rule)
                    result['success'] = True
        
        if result['success']:
            for rule in result['matched_rules']:
                if 'products' in rule:
                    result['products'].extend(rule['products'])
                result['mechanism'] = rule.get('mechanism', '')
                result['conditions'] = rule.get('conditions', '')
                if 'notes' in rule:
                    result['notes'].extend([rule['notes']] if isinstance(rule['notes'], str) else rule['notes'])
        
        # Remove duplicates
        result['products'] = list(set(result['products']))
        
        return result
    
    def add_custom_reaction(self, reaction_data: Dict):
        """Add a custom reaction rule"""
        reaction_type = reaction_data.get('type')
        if reaction_type not in self.reaction_rules:
            self.reaction_rules[reaction_type] = []
        
        # Generate unique ID for the reaction
        reaction_id = hashlib.md5(
            json.dumps(reaction_data, sort_keys=True).encode()
        ).hexdigest()[:8]
        reaction_data['id'] = reaction_id
        
        self.reaction_rules[reaction_type].append(reaction_data)
        return reaction_id
    
    def get_reaction_types(self) -> List[str]:
        """Get all available reaction types"""
        return list(self.reaction_rules.keys())
    
    def get_all_reactions(self) -> Dict:
        """Get all reaction rules"""
        return self.reaction_rules

# Initialize the engine
reaction_engine = ChemistryReactionEngine()

# ----------------------------
# CORE CHEMISTRY FUNCTIONS
# ----------------------------

def get_compound_database():
    """Get compound database from engine"""
    return reaction_engine.get_compound_database()

def get_compound_name(smiles):
    """Get compound name from SMILES"""
    # First fix the SMILES
    fixed_smiles = reaction_engine.fix_smiles(smiles)
    
    # Look in database
    database = get_compound_database()
    for name, smi in database.items():
        if smi == fixed_smiles:
            return name.title()
    
    # Try to generate descriptive name
    try:
        mol = Chem.MolFromSmiles(fixed_smiles)
        if mol:
            groups = reaction_engine.get_functional_groups(fixed_smiles)
            if groups:
                if 'phenol' in groups:
                    return "Phenol derivative"
                elif 'carboxylic_acid' in groups and 'aryl' in groups:
                    return "Aromatic carboxylic acid"
                elif 'aldehyde' in groups and 'aryl' in groups:
                    return "Aromatic aldehyde"
                elif 'ketone' in groups and 'aryl' in groups:
                    return "Aromatic ketone"
                elif 'amine_primary' in groups and 'aryl' in groups:
                    return "Aromatic amine"
                elif 'nitro' in groups and 'aryl' in groups:
                    return "Nitroaromatic compound"
                elif 'alcohol_primary' in groups:
                    return "Primary alcohol"
                elif 'alcohol_secondary' in groups:
                    return "Secondary alcohol"
                elif 'ester' in groups:
                    return "Ester"
                elif 'amide' in groups:
                    return "Amide"
    except:
        pass
    
    return "Unknown compound"

def validate_smiles(smiles):
    """Check if SMILES is valid"""
    try:
        fixed = reaction_engine.fix_smiles(smiles)
        mol = Chem.MolFromSmiles(fixed)
        return mol is not None
    except:
        return False

def draw_molecule(smiles, size=(200, 200)):
    """Draw a molecule from SMILES"""
    try:
        fixed = reaction_engine.fix_smiles(smiles)
        mol = Chem.MolFromSmiles(fixed)
        if mol:
            return Draw.MolToImage(mol, size=size)
    except:
        return None
    return None

def calculate_molecular_properties(smiles):
    """Calculate molecular properties"""
    try:
        fixed = reaction_engine.fix_smiles(smiles)
        mol = Chem.MolFromSmiles(fixed)
        if mol:
            props = {
                'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} g/mol",
                'Formula': rdMolDescriptors.CalcMolFormula(mol),
                #'IUPAC NAME':rdMolDescriptors.iupac_name(mol),
                'Heavy Atoms': mol.GetNumHeavyAtoms(),
                'Total Atoms': mol.GetNumAtoms(),
                'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                'H-Bond Donors': Descriptors.NumHDonors(mol),
                'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
                'LogP': f"{Descriptors.MolLogP(mol):.2f}",
                'TPSA': f"{Descriptors.TPSA(mol):.2f} Å²",
                'MR': f"{Descriptors.MolMR(mol):.2f}",
            }
            
            # Add functional groups info
            groups = reaction_engine.get_functional_groups(fixed)
            if groups:
                props['Functional Groups'] = ', '.join(sorted(groups)[:5])
            
            return props
    except:
        return {}
    return None

# ----------------------------
# LLM FUNCTIONS
# ----------------------------

# --- Load secrets early ---
def get_api_key():
    # Prefer secrets; fallback to environment
    return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

api_key = get_api_key()

# Optional: export to environment for libraries that read it
os.environ["GROQ_API_KEY"] = api_key

# --- LLM config (uses internal key only) ---
LLM_CONFIG = {
    "api_url": "https://api.groq.com/openai/v1/chat/completions",
    "default_model": "llama-3.1-8b-instant",
    "api_key": api_key,
}

# Initialize session state
st.session_state.setdefault("llm_messages", [])

def call_llm(prompt, api_key=None, model=None, system_prompt=None):
    """Call Groq (OpenAI-compatible) LLM API via HTTP POST"""
    if not api_key:
        return "⚠ Please provide an API key in the settings or Secrets."
    
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
        
        # Try to parse JSON safely
        try:
            data = response.json()
        except Exception:
            return f"❌ API returned non-JSON response (status {response.status_code}): {response.text}"
        
        if response.status_code == 200:
            # handle common OpenAI-compatible response shapes
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                # fallback: return a stringified JSON for debugging
                return json.dumps(data, indent=2)
        else:
            return f"❌ API Error {response.status_code}: {response.text}"
    
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

def chemistry_expert_system_prompt():
    """System prompt for chemistry expert LLM"""
    return ("You are an expert organic chemistry professor with 20+ years of experience.\n\n"
            "Your expertise includes synthesis, mechanisms, spectroscopy, and safety.\n\n"
            "Guidelines: be accurate, use IUPAC where possible, provide stepwise mechanisms, and include SMILES when relevant.")

def analyze_with_llm(reaction_info, api_key, model):
    """Send reaction analysis to LLM for expert explanation"""
    prompt = f"""As a chemistry expert, analyze this reaction:\n\nREACTION INFORMATION:\n{json.dumps(reaction_info, indent=2)}\n\nPlease provide a detailed mechanism, alternative routes, experimental considerations, side reactions, and safety notes."""
    return call_llm(prompt, api_key, model, chemistry_expert_system_prompt())

# ----------------------------
# ENHANCED SIDEBAR
# ----------------------------

def render_enhanced_sidebar():
    """Enhanced sidebar with comprehensive compound database"""
    with st.sidebar:
        st.title("⚗️ Chemistry AI Assistant")
        st.markdown("---")
        
        # App mode selection
        st.session_state.app_mode = st.radio(
            "Select Mode:",
            ["Enhanced Chemistry Solver", "AI Chemistry Assistant", "LLM Chat", "Reaction Database", "Compound Explorer"],
            index=0
        )
        
        st.markdown("---")
        
        # LLM Settings
        if st.session_state.app_mode in ["AI Chemistry Assistant", "LLM Chat"]:
            st.subheader("🤖 LLM Configuration")
            
            # Populate API key input with secret value if present; keep it masked
            api_key_value = LLM_CONFIG.get("api_key", "")
            user_input_key = st.text_input(
                "Groq API Key",
                value=api_key_value,
                type="password",
                help="Get your API key from https://console.groq.com"
            )
            # final effective key: preference to secret, then user input
            if api_key_value:
                LLM_CONFIG["api_key"] = api_key_value
            else:
                LLM_CONFIG["api_key"] = user_input_key
            
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
            if st.button("Clear Chat History"):
                st.session_state.llm_messages = []
                st.rerun()
        
        # Quick tools
        st.subheader("🔧 Quick Tools")
        
        with st.expander("🧪 Compound Lookup", expanded=True):
            # Get compounds by category
            compound_groups = reaction_engine.get_compound_groups()
            
            # Category selector
            category = st.selectbox(
                "Select Compound Category:",
                list(compound_groups.keys())
            )
            
            if category in compound_groups and compound_groups[category]:
                # Compound selector within category
                compound_names = list(compound_groups[category].keys())
                selected_compound = st.selectbox(
                    f"Select {category[:-1] if category.endswith('s') else category}:",
                    compound_names
                )
                
                if selected_compound:
                    smiles = compound_groups[category][selected_compound]
                    st.code(smiles)
                    
                    # Show molecule
                    img = draw_molecule(smiles, (120, 120))
                    if img:
                        st.image(img, caption=selected_compound.title())
                    
                    # Quick properties
                    props = calculate_molecular_properties(smiles)
                    if props:
                        st.caption(f"MW: {props.get('Molecular Weight', 'N/A')}")
                        st.caption(f"Formula: {props.get('Formula', 'N/A')}")
        
        with st.expander("🔍 Functional Group Detector", expanded=False):
            fg_smiles = st.text_input("Analyze SMILES:", "c1ccccc1C(=O)O")
            if fg_smiles:
                groups = reaction_engine.get_functional_groups(fg_smiles)
                if groups:
                    st.write("**Detected functional groups:**")
                    for g in groups:
                        st.write(f"- {g.replace('_', ' ').title()}")
                else:
                    st.info("No functional groups detected")
        
        with st.expander("✅ SMILES Validator", expanded=False):
            test_smiles = st.text_input("Test SMILES:", "Oc1ccccc1")
            if test_smiles:
                fixed = reaction_engine.fix_smiles(test_smiles)
                if validate_smiles(fixed):
                    st.success("✅ Valid SMILES")
                    st.code(fixed)
                else:
                    st.error("❌ Invalid SMILES")
        
        st.markdown("---")
        st.markdown("[help file](https://github.com/ramj39/AskLLM/blob/main/README.md)")
        st.markdown("[reference literature](https://en.wikipedia.org/wiki/List_of_organic_reactions)") 
        st.caption("Developed by Subramanian Ramajayam")

# ----------------------------
# REACTION DATABASE VIEW
# ----------------------------

def render_reaction_database():
    """Browse all available reactions with comprehensive display"""
    st.title("📚 Reaction Database")
    st.markdown("Browse all available reaction rules in the system")
    
    # Get all reactions
    all_reactions = reaction_engine.get_all_reactions()
    
    # Statistics
    total_rules = sum(len(rules) for rules in all_reactions.values())
    st.metric("Total Reaction Rules", total_rules)
    
    # Search and filter
    col_search, col_filter, col_stats = st.columns([2, 1, 1])
    
    with col_search:
        search_term = st.text_input("🔍 Search reactions by name, mechanism, or conditions:", "")
    
    with col_filter:
        selected_type = st.selectbox(
            "Filter by reaction type:",
            ["All"] + list(all_reactions.keys())
        )
    
    with col_stats:
        st.metric("Reaction Types", len(all_reactions))
    
    st.markdown("---")
    
    # If no reactions found, show message
    if not all_reactions:
        st.warning("No reaction rules found in the database.")
        return
    
    # Display reactions
    reaction_count = 0
    
    for rxn_type, rules in all_reactions.items():
        if selected_type != "All" and rxn_type != selected_type:
            continue
        
        # Filter by search term
        filtered_rules = []
        for rule in rules:
            if not search_term or search_term.lower() in str(rule).lower():
                filtered_rules.append(rule)
        
        if not filtered_rules:
            continue
        
        with st.expander(f"**{rxn_type.replace('_', ' ').title()}** ({len(filtered_rules)} reactions)", expanded=True):
            for i, rule in enumerate(filtered_rules):
                reaction_count += 1
                
                # Create columns for layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"##### {rule.get('name', f'Reaction {i+1}')}")
                    
                    # Display reactants and products
                    if 'reactant_patterns' in rule:
                        st.write(f"**Reactants**: {', '.join(rule['reactant_patterns'])}")
                    if 'products' in rule:
                        st.write(f"**Products**: {', '.join(rule['products'])}")
                    
                    # Display mechanism
                    if 'mechanism' in rule:
                        st.caption(f"*Mechanism*: {rule['mechanism']}")
                    
                    # Display conditions
                    if 'conditions' in rule:
                        st.info(f"**Conditions**: {rule['conditions']}")
                    
                    # Display reagents and catalyst
                    reagents_text = ""
                    if 'reagents' in rule and rule['reagents']:
                        reagents_text += f"**Reagents**: {', '.join(rule['reagents'])}"
                    if 'catalyst' in rule and rule['catalyst']:
                        if reagents_text:
                            reagents_text += " | "
                        reagents_text += f"**Catalyst**: {', '.join(rule['catalyst'])}"
                    
                    if reagents_text:
                        st.write(reagents_text)
                
                with col2:
                    # Visual representation
                    if 'products' in rule and rule['products']:
                        # Try to draw the first product
                        product_smiles = rule['products'][0]
                        if validate_smiles(product_smiles):
                            img = draw_molecule(product_smiles, (150, 150))
                            if img:
                                st.image(img, caption="Product Structure")
                    
                    # Use button
                    if st.button("Use This Reaction", key=f"use_{rxn_type}_{i}"):
                        st.session_state.app_mode = "Enhanced Chemistry Solver"
                        st.rerun()
                
                st.markdown("---")
    
    if reaction_count == 0 and search_term:
        st.info(f"No reactions found matching '{search_term}'")

# ----------------------------
# COMPOUND EXPLORER
# ----------------------------

def render_compound_explorer():
    """Comprehensive compound explorer"""
    st.title("🔬 Compound Explorer")
    st.markdown("Explore and analyze chemical compounds")
    
    # Get compound database
    database = get_compound_database()
    compound_groups = reaction_engine.get_compound_groups()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Browse by Category", "Search Compounds", "Advanced Analysis"])
    
    with tab1:
        st.subheader("Browse Compounds by Functional Group")
        
        # Category selector
        categories = list(compound_groups.keys())
        selected_category = st.selectbox("Select Category:", categories)
        
        if selected_category in compound_groups:
            compounds = compound_groups[selected_category]
            
            if compounds:
                # Display compounds in the category
                num_cols = 3
                cols = st.columns(num_cols)
                
                for idx, (name, smiles) in enumerate(compounds.items()):
                    col_idx = idx % num_cols
                    with cols[col_idx]:
                        # Card-like display
                        with st.container():
                            st.markdown(f"**{name.title()}**")
                            st.code(smiles)
                            
                            # Draw molecule
                            img = draw_molecule(smiles, (120, 120))
                            if img:
                                st.image(img)
                            
                            # Quick actions
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Use", key=f"use_{name}"):
                                    # Store for use in solver
                                    st.session_state.selected_compound = smiles
                                    st.session_state.app_mode = "Enhanced Chemistry Solver"
                                    st.rerun()
                            with col2:
                                if st.button("Analyze", key=f"analyze_{name}"):
                                    # Show properties
                                    st.session_state.analyze_compound = smiles
                
                # Show compound count
                st.info(f"Found {len(compounds)} compounds in {selected_category}")
            else:
                st.warning(f"No compounds found in {selected_category}")
    
    with tab2:
        st.subheader("Search Compounds")
        
        search_query = st.text_input("Search by name or properties:", "")
        
        if search_query:
            # Search in database
            results = []
            for name, smiles in database.items():
                if (search_query.lower() in name.lower() or 
                    search_query in smiles):
                    results.append((name, smiles))
            
            if results:
                st.write(f"Found {len(results)} compounds:")
                
                # Display results
                for name, smiles in results[:20]:  # Limit to 20 results
                    with st.expander(name.title()):
                        st.code(smiles)
                        img = draw_molecule(smiles, (100, 100))
                        if img:
                            st.image(img)
                        
                        # Quick properties
                        props = calculate_molecular_properties(smiles)
                        if props:
                            st.write(f"**Molecular Weight**: {props.get('Molecular Weight', 'N/A')}")
                            st.write(f"**Formula**: {props.get('Formula', 'N/A')}")
            else:
                st.warning(f"No compounds found matching '{search_query}'")
    
    with tab3:
        st.subheader("Advanced Compound Analysis")
        
        # Input SMILES for analysis
        analysis_smiles = st.text_input("Enter SMILES for detailed analysis:", "c1ccccc1C(=O)O")
        
        if analysis_smiles and validate_smiles(analysis_smiles):
            fixed_smiles = reaction_engine.fix_smiles(analysis_smiles)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Structure")
                img = draw_molecule(fixed_smiles, (300, 300))
                if img:
                    st.image(img, caption=get_compound_name(fixed_smiles))
            
            with col2:
                st.subheader("Properties")
                props = calculate_molecular_properties(fixed_smiles)
                if props:
                    for key, value in props.items():
                        st.write(f"**{key}**: {value}")
                
                st.subheader("Functional Groups")
                groups = reaction_engine.get_functional_groups(fixed_smiles)
                if groups:
                    for group in groups:
                        st.write(f"• {group.replace('_', ' ').title()}")
                else:
                    st.info("No functional groups detected")
        
        else:
            st.warning("Please enter a valid SMILES notation")

# ----------------------------
# ENHANCED CHEMISTRY SOLVER
# ----------------------------

def render_enhanced_chemistry_solver():
    """Enhanced version with better reaction prediction"""
    st.title("🧪 Enhanced Chemistry Problem Solver")
    st.markdown("""
    Solve organic chemistry problems with comprehensive reaction prediction.
    The system now includes:
    - **Pattern-based reaction matching**
    - **Functional group detection**
    - **Mechanism explanations**
    - **Reaction conditions**
    """)
    
    # Two-column layout
    col_config, col_info = st.columns([1, 1])
    
    with col_config:
        st.subheader("⚙️ Reaction Configuration")
        
        # Number of reactants
        num_reactants = st.slider("Number of reactants:", 1, 5, 2, 
                                 help="Select how many reactants you have")
        
        # Reaction type selection with descriptions
        reaction_types = reaction_engine.get_reaction_types()
        reaction_descriptions = {
            'nitration': 'Add NO₂ group to aromatic rings',
            'oxidation': 'Increase oxygen/decrease hydrogen',
            'reduction': 'Decrease oxygen/increase hydrogen',
            'esterification': 'Acid + Alcohol → Ester',
            'hydrolysis': 'Cleavage with water',
            'acetylation': 'Add acetyl group',
            'halogenation': 'Add halogen atom',
            'ammonolysis': 'Reaction with ammonia',
            'friedel_crafts_alkylation': 'Alkylation of aromatics',
            'sulfonation': 'Add SO₃H group',
            'grignard': 'Organometallic reactions',
        }
        
        selected_type = st.selectbox(
            "Select reaction type:",
            reaction_types,
            format_func=lambda x: f"{x.title()} - {reaction_descriptions.get(x, 'General reaction')}"
        )
        
        # Quick examples
        example_systems = {
            'Phenol nitration': ['Oc1ccccc1', 'O=N(=O)O'],
            'Benzene nitration': ['c1ccccc1', 'O=N(=O)O'],
            'Esterification': ['c1ccccc1C(=O)O', 'CO'],
            'Reduction of nitrobenzene': ['O=[N+]([O-])c1ccccc1'],
        }
        
        example = st.selectbox(
            "Load example system:",
            ["None"] + list(example_systems.keys())
        )
    
    with col_info:
        st.subheader("ℹ️ Reaction Info")
        if selected_type in reaction_descriptions:
            st.info(reaction_descriptions[selected_type])
        
        # Show available reactions for this type
        rules = reaction_engine.reaction_rules.get(selected_type, [])
        if rules:
            with st.expander(f"Available {selected_type} reactions ({len(rules)})"):
                for rule in rules[:3]:
                    st.markdown(f"**{rule['name']}**")
                    st.caption(f"Reactants: {', '.join(rule.get('reactant_patterns', ['Various']))}")
                    st.caption(f"Conditions: {rule.get('conditions', 'Various')}")
        
        # Functional group helper
        st.subheader("🔍 Functional Group Detector")
        test_smiles = st.text_input("Test SMILES for functional groups:", "Oc1ccccc1")
        if test_smiles and validate_smiles(test_smiles):
            groups = reaction_engine.get_functional_groups(test_smiles)
            if groups:
                st.success(f"Found: {', '.join(groups)}")
    
    st.markdown("---")
    
    # Reactant input section
    st.subheader("🧪 Reactant Input")
    reactants = []
    
    # Use example if selected
    if example != "None":
        default_reactants = example_systems[example]
        num_reactants = len(default_reactants)
    
    # Create input columns
    cols = st.columns(min(num_reactants, 3))
    default_examples = ['Oc1ccccc1', 'O=N(=O)O', 'c1ccccc1C(=O)O', 'CO', 'Nc1ccccc1']
    
    for i in range(num_reactants):
        col_idx = i % 3
        with cols[col_idx]:
            # Set default from example or sequence
            if example != "None" and i < len(example_systems[example]):
                default = example_systems[example][i]
            elif i < len(default_examples):
                default = default_examples[i]
            else:
                default = ""
            
            r_smiles = st.text_input(
                f"Reactant {i+1} SMILES", 
                default,
                key=f"reactant_{i}",
                help="Enter SMILES notation or common name"
            )
            
            if r_smiles:
                # Handle common names
                db = get_compound_database()
                if r_smiles.lower() in db:
                    r_smiles = db[r_smiles.lower()]
                
                # Validate and fix
                fixed_smiles = reaction_engine.fix_smiles(r_smiles)
                if validate_smiles(fixed_smiles):
                    reactants.append(fixed_smiles)
                    st.success(f"✅ {get_compound_name(fixed_smiles)}")
                    
                    # Show molecule
                    img = draw_molecule(fixed_smiles, (120, 120))
                    if img:
                        st.image(img, caption=f"R{i+1}")
                    
                    # Show functional groups
                    groups = reaction_engine.get_functional_groups(fixed_smiles)
                    if groups:
                        st.caption(f"Groups: {', '.join(groups[:3])}")
                else:
                    st.error("Invalid SMILES")
                    # Try to suggest fix
                    fixed = reaction_engine.fix_smiles(r_smiles)
                    if fixed != r_smiles and validate_smiles(fixed):
                        st.info(f"Try: {fixed}")
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🔬 Analyze Reaction", type="primary"):
        if not reactants:
            st.warning("Please enter at least one valid reactant!")
        else:
            with st.spinner("Analyzing reaction with enhanced engine..."):
                # Use enhanced prediction
                result = reaction_engine.predict_reaction(reactants, selected_type)
                
                # Display results
                st.subheader("📊 Reaction Analysis Results")
                
                if result['success']:
                    st.success("✅ Reaction predicted successfully!")
                    
                    # Display reactants
                    st.markdown("### Reactants")
                    r_cols = st.columns(len(reactants))
                    for idx, (col, r_smiles) in enumerate(zip(r_cols, reactants)):
                        with col:
                            st.markdown(f"**Reactant {idx+1}**")
                            st.code(r_smiles)
                            st.write(get_compound_name(r_smiles))
                            img = draw_molecule(r_smiles, (100, 100))
                            if img:
                                st.image(img)
                    
                    # Display products
                    st.markdown("### Predicted Products")
                    if result['products']:
                        p_cols = st.columns(min(len(result['products']), 4))
                        for idx, (col, p_smiles) in enumerate(zip(p_cols, result['products'])):
                            with col:
                                st.markdown(f"**Product {idx+1}**")
                                st.code(p_smiles)
                                st.write(get_compound_name(p_smiles))
                                img = draw_molecule(p_smiles, (100, 100))
                                if img:
                                    st.image(img)
                        
                        # Show reaction equation
                        st.markdown("### Reaction Equation")
                        reactants_str = " + ".join([get_compound_name(r) for r in reactants])
                        products_str = " + ".join([get_compound_name(p) for p in result['products']])
                        st.markdown(f"**{reactants_str} → {products_str}**")
                    else:
                        st.warning("No specific products predicted")
                    
                    # Display mechanism and conditions
                    st.markdown("### Reaction Details")
                    col1, col2 = st.columns(2)
                    with col1:
                            reagents = st.text_area("Reagents (one per line)", "Reagent1\nReagent2")
                            catalyst = st.text_input("Catalyst", "H₂SO₄")
                    with col2:
                            temperature = st.text_input("Temperature", "Room temperature")
                            solvent = st.text_input("Solvent", "None")
            
            mechanism = st.text_area("Mechanism", "Step-by-step mechanism...")
            
            # Submit
            if st.button("Add Custom Reaction",key="add_custom_btn"):
            #submitted = st.form_submit_button("Add Custom Reaction")
            
            #if submitted:
                # Validate inputs
                valid = True
                for r in reactants_custom + products_custom:
                    if r and not validate_smiles(r):
                        st.error(f"Invalid SMILES: {r}")
                        valid = False
                
                if valid:
                    # Create reaction data
                    reaction_data = {
                        'type': reaction_type,
                        'name': reaction_name,
                        'description': description,
                        'reactants': reactants_custom,
                        'products': products_custom,
                        'reagents': [r.strip() for r in reagents.split('\n') if r.strip()],
                        'catalyst': [catalyst] if catalyst else [],
                        'conditions': f"{temperature}, {solvent}",
                        'mechanism': mechanism,
                        'min_reactants': len([r for r in reactants_custom if r]),
                        'max_reactants': len([r for r in reactants_custom if r]),
                    }
                    
                    # Add to engine
                    reaction_id = reaction_engine.add_custom_reaction(reaction_data)
                    st.success(f"✅ Reaction added successfully! (ID: {reaction_id})")
                    
                    # Show preview
                    with st.expander("Preview Reaction"):
                        st.json(reaction_data)
                    # ----------------------------
# AI ASSISTANT
# ----------------------------

def render_ai_assistant():
    """AI Chemistry Assistant"""
    st.title("AI Chemistry Assistant")
    st.markdown("Get expert explanations for chemistry problems using AI")
    
    if not LLM_CONFIG["api_key"]:
        st.warning("Please provide your Groq API key in the sidebar or add it to Streamlit Secrets as GROQ_API_KEY.")
        return
    
    col_chem, col_llm = st.columns([1, 1])
    
    with col_chem:
        st.subheader("Chemistry Input")
        reaction_input = st.text_area("Describe a reaction or ask a chemistry question:", height=150)
        with st.expander("Or use structured input"):
            num_reactants = st.slider("Reactants", 1, 3, 1, key="ai_reactants")
            reactants = []
            for i in range(num_reactants):
                r = st.text_input(f"Reactant {i+1}", key=f"ai_reactant_{i}")
                if r:
                    reactants.append(r)
            reaction_type = st.selectbox("Reaction Type", ["oxidation", "reduction", "esterification", "hydrolysis", "acetylation", "halogenation", "nitration", "ammonolysis"], key="ai_reaction_type")
        
        if st.button("Analyze with AI"):
            if reaction_input or reactants:
                with st.spinner("Consulting chemistry expert..."):
                    reaction_info = {
                        "user_query": reaction_input,
                        "reactants": reactants if reactants else [],
                        "reaction_type": reaction_type if 'reaction_type' in locals() else None,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    llm_response = analyze_with_llm(reaction_info, LLM_CONFIG["api_key"], LLM_CONFIG["default_model"])
                    st.session_state.llm_messages.append({"role": "user", "content": reaction_input or f"Analyze: {reactants} -> {reaction_type}"})
                    st.session_state.llm_messages.append({"role": "assistant", "content": llm_response})
    
    with col_llm:
        st.subheader("AI Response")
        for msg in st.session_state.llm_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                 with st.chat_message("assistant"):
                    st.markdown(msg["content"])
        
        if st.session_state.llm_messages:
            chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.llm_messages])
            st.download_button(label="Download Conversation", data=chat_text, file_name="chemistry_ai_chat.txt", mime="text/plain")

# ----------------------------
# LLM CHAT
# ----------------------------

def render_llm_chat():
    """General LLM Chat"""
    st.title("General LLM Chat")
    
    if not LLM_CONFIG["api_key"]:
        st.warning("Please provide your Groq API key in the sidebar or add it to Streamlit Secrets as GROQ_API_KEY.")
        return
    
    user_input = st.text_area("Enter your message:", height=100)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Send Message"):
            if user_input:
                with st.spinner("Thinking..."):
                    st.session_state.llm_messages.append({"role": "user", "content": user_input})
                    response = call_llm(user_input, LLM_CONFIG["api_key"], LLM_CONFIG["default_model"])
                    st.session_state.llm_messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.llm_messages = []
            st.rerun()
    
    st.markdown("---")
    st.subheader("Conversation History")
    for msg in st.session_state.llm_messages:
        if msg["role"] == "user":
            st.markdown(f"You: {msg['content']}")
        else:
            st.markdown(f"Assistant: {msg['content']}")
        st.markdown("---")
    
    if st.session_state.llm_messages:
        chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.llm_messages])
        st.download_button(label="Download Full Chat", data=chat_text, file_name="llm_chat_history.txt", mime="text/plain", use_container_width=True)

# ----------------------------
# MAIN APP FUNCTION
# ----------------------------

def main():
    st.set_page_config(
        page_title="Enhanced Chemistry AI Assistant",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session states
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Enhanced Chemistry Solver"
    
    # Render enhanced sidebar
    render_enhanced_sidebar()
    
    # Main content routing
    if st.session_state.app_mode == "Enhanced Chemistry Solver":
        render_enhanced_chemistry_solver()
    
    elif st.session_state.app_mode == "Reaction Database":
        render_reaction_database()
    
    elif st.session_state.app_mode == "Compound Explorer":
        render_compound_explorer()
    
    elif st.session_state.app_mode == "AI Chemistry Assistant":
        render_ai_assistant()
    
    elif st.session_state.app_mode == "LLM Chat":
        render_llm_chat()

# ----------------------------
# RUN THE APP
# ----------------------------
st.markdown("---")
st.header("💬 Feedback")

feedback = st.text_area("Share your comments or suggestions here:")

if st.button("Submit Feedback"):
    if feedback.strip():
        st.success("✅ Thanks for your feedback! We appreciate your input.")
        # Optional: save feedback to a file
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")
    else:
        st.warning("⚠️ Please enter some feedback before submitting.")

import sqlite3

conn = sqlite3.connect("visitors.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS counter (visits INTEGER)")
c.execute("SELECT COUNT(*) FROM counter")
if c.fetchone()[0] == 0:
    c.execute("INSERT INTO counter VALUES (0)")
    conn.commit()

c.execute("UPDATE counter SET visits = visits + 1")
conn.commit()

c.execute("SELECT visits FROM counter")
count = c.fetchone()[0]

st.sidebar.write(f"👥 Total visitors: {count}")


if __name__ == "__main__":
    main()











