import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import os
from datetime import datetime

from src.config import (
    CLINVAR_CLEAN_FILE, GWAS_CLEAN_FILE,
    HPO_DISEASE_PHENOTYPE_FILE, HPO_GENE_PHENOTYPE_FILE,
    RELATIONSHIPS_FILE, ENTITY_SUMMARY_CSV, DATASET_PROFILE_FILE
)
from src.ml.dna_verifier import verify_dna_sequence
from src.ml.classifier import predict_and_justify, train_and_evaluate_model

st.set_page_config(page_title="HUMAN OS", layout="wide", page_icon="🧬")

# Custom CSS for the "futuristic biomedical interface" theme
st.markdown("""
<style>
    :root {
        --primary-bg: #0e1117;
        --secondary-bg: #262730;
        --cyan-accent: #00e5ff;
        --blue-accent: #2979ff;
        --green-accent: #00e676;
        --warm-accent: #ff9100;
        --text-color: #f5f5f5;
    }
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-color);
    }
    h1, h2, h3, h4 {
        color: var(--cyan-accent) !important;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 1px;
    }
    .kpi-card {
        background: var(--secondary-bg);
        border-left: 4px solid var(--blue-accent);
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--green-accent);
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background: #000;
        color: #888;
        font-size: 0.8rem;
    }
    /* Hide some default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_data():
    data = {}
    try:
        if CLINVAR_CLEAN_FILE.exists():
            data['clinvar'] = pd.read_parquet(CLINVAR_CLEAN_FILE)
        else:
            data['clinvar'] = pd.DataFrame()
            
        if GWAS_CLEAN_FILE.exists():
            data['gwas'] = pd.read_parquet(GWAS_CLEAN_FILE)
        else:
            data['gwas'] = pd.DataFrame()
            
        if HPO_DISEASE_PHENOTYPE_FILE.exists():
            data['hpo_dis'] = pd.read_parquet(HPO_DISEASE_PHENOTYPE_FILE)
        else:
            data['hpo_dis'] = pd.DataFrame()
            
        if HPO_GENE_PHENOTYPE_FILE.exists():
            data['hpo_gene'] = pd.read_parquet(HPO_GENE_PHENOTYPE_FILE)
        else:
            data['hpo_gene'] = pd.DataFrame()
            
        if RELATIONSHIPS_FILE.exists():
            data['rels'] = pd.read_parquet(RELATIONSHIPS_FILE)
        else:
            data['rels'] = pd.DataFrame()
            
        if ENTITY_SUMMARY_CSV.exists():
            data['summary'] = pd.read_csv(ENTITY_SUMMARY_CSV)
        else:
            data['summary'] = pd.DataFrame()

        ml_seq_file = Path(__file__).resolve().parent / "data" / "processed" / "ml_sequences.parquet"
        if ml_seq_file.exists():
            data['ml_seq'] = pd.read_parquet(ml_seq_file)
        else:
            data['ml_seq'] = pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return data

data = load_data()

# Navigation
st.sidebar.title("HUMAN OS")
st.sidebar.markdown("### Interactive Genomic Variant and Disease Visualization")

pages = [
    "Overview", 
    "Variant Explorer", 
    "Gene / Phenotype Explorer", 
    "Disease / Phenotype Explorer", 
    "GWAS Explorer", 
    "Network Explorer",
    "Interactive Chart Studio",
    "Data Sources & Methodology",
    "DNA Sequence Disease Classifier"
]

# Global Search
search_query = st.sidebar.text_input("Global Search", placeholder="e.g. BRCA1, HP:0000001, rs12345")
if search_query:
    # Just redirecting context for now
    st.sidebar.info(f"Searching for: {search_query}. Navigating to appropriate explorer is suggested based on query type.")

selection = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #888;">
Educational data visualization. Not medical advice.
</div>
""", unsafe_allow_html=True)

# Helper for download
def get_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    return st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

def render_dynamic_chart(df, x_col, y_col, title, default_type="Bar (Vertical)", key="", height=340, color_sequence=None):
    """
    Renders an interactive chart with a dynamic graph type selector,
    allowing the user to switch between Bar, Horizontal Bar, Donut, Line, Area, Treemap, and Funnel.
    """
    if df is None or df.empty:
        st.info(f"No data available for {title}")
        return

    chart_types = ["Bar (Vertical)", "Bar (Horizontal)", "Pie / Donut", "Line", "Area", "Treemap", "Funnel"]
    default_idx = chart_types.index(default_type) if default_type in chart_types else 0
    
    col_hdr, col_sel = st.columns([3, 2])
    with col_hdr:
        st.subheader(title)
    with col_sel:
        chosen_type = st.selectbox(
            f"Graph Type",
            chart_types,
            index=default_idx,
            key=f"dyn_chart_{key}_{x_col}_{y_col}",
            label_visibility="collapsed"
        )
        
    palette = color_sequence or ['#00e5ff', '#2979ff', '#00e676', '#ff9100', '#d500f9', '#ffd600', '#ff1744', '#00b0ff']
    fig = None
    
    if chosen_type == "Bar (Vertical)":
        fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark", color_discrete_sequence=palette)
    elif chosen_type == "Bar (Horizontal)":
        fig = px.bar(df, x=y_col, y=x_col, orientation='h', template="plotly_dark", color_discrete_sequence=palette)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    elif chosen_type == "Pie / Donut":
        fig = px.pie(df, names=x_col, values=y_col, template="plotly_dark", hole=0.4, color_discrete_sequence=palette)
    elif chosen_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, markers=True, template="plotly_dark", color_discrete_sequence=palette)
    elif chosen_type == "Area":
        fig = px.area(df, x=x_col, y=y_col, template="plotly_dark", color_discrete_sequence=palette)
    elif chosen_type == "Treemap":
        fig = px.treemap(df, path=[x_col], values=y_col, template="plotly_dark", color_discrete_sequence=palette)
    elif chosen_type == "Funnel":
        fig = px.funnel(df, x=y_col, y=x_col, template="plotly_dark", color_discrete_sequence=palette)

    if fig:
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=25, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_overview():
    st.title("HUMAN OS Overview")
    
    if data['summary'].empty:
        st.warning("Data not prepared yet. Please run data preparation scripts.")
        return
        
    summary_dict = dict(zip(data['summary']['entity_type'], data['summary']['count']))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='kpi-card'>Variants<br><span class='kpi-value'>{summary_dict.get('Variants', 0)}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi-card'>Genes<br><span class='kpi-value'>{summary_dict.get('Genes', 0)}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi-card'>Diseases<br><span class='kpi-value'>{summary_dict.get('Diseases', 0)}</span></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='kpi-card'>Phenotypes<br><span class='kpi-value'>{summary_dict.get('Phenotypes', 0)}</span></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='kpi-card'>GWAS Traits<br><span class='kpi-value'>{summary_dict.get('Traits', 0)}</span></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    colA, colB = st.columns(2)
    
    with colA:
        if not data['clinvar'].empty:
            chr_counts = data['clinvar']['chromosome'].value_counts().reset_index()
            chr_counts.columns = ['Chromosome', 'Count']
            render_dynamic_chart(chr_counts, 'Chromosome', 'Count', "Variant Count by Chromosome", default_type="Bar (Vertical)", key="ov_chr", color_sequence=['#2979ff'])
            
    with colB:
        if not data['clinvar'].empty:
            sig_counts = data['clinvar']['clinical_significance'].value_counts().reset_index()
            sig_counts.columns = ['Significance', 'Count']
            render_dynamic_chart(sig_counts, 'Significance', 'Count', "ClinVar Clinical Significance", default_type="Pie / Donut", key="ov_sig")
            
    colC, colD = st.columns(2)
    
    with colC:
        if not data['clinvar'].empty:
            gene_counts = data['clinvar']['gene'].value_counts().head(10).reset_index()
            gene_counts.columns = ['Gene', 'Count']
            render_dynamic_chart(gene_counts, 'Gene', 'Count', "Top Genes by ClinVar Records", default_type="Bar (Horizontal)", key="ov_gene", color_sequence=['#00e676'])
            
    with colD:
        if not data['gwas'].empty:
            trait_counts = data['gwas']['trait'].value_counts().head(10).reset_index()
            trait_counts.columns = ['Trait', 'Count']
            render_dynamic_chart(trait_counts, 'Trait', 'Count', "Top GWAS Traits", default_type="Bar (Horizontal)", key="ov_trait", color_sequence=['#ff9100'])

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    - Human Phenotype Ontology (HPO)
    - NCBI ClinVar
    - NHGRI-EBI GWAS Catalog
    """)
    if DATASET_PROFILE_FILE.exists():
        stat = os.stat(DATASET_PROFILE_FILE)
        st.markdown(f"*Last local data refresh: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}*")


def show_variant_explorer():
    st.title("Variant Explorer")
    if data['clinvar'].empty:
        st.warning("ClinVar data has not yet been prepared. Run the refresh script.")
        return
        
    df = data['clinvar'].copy()
    
    st.sidebar.subheader("Filters")
    genes = ["All"] + sorted(df['gene'].dropna().unique().tolist())
    sel_gene = st.sidebar.selectbox("Gene", genes)
    
    chrs = ["All"] + sorted(df['chromosome'].dropna().unique().astype(str).tolist())
    sel_chr = st.sidebar.selectbox("Chromosome", chrs)
    
    sigs = ["All"] + sorted(df['clinical_significance'].dropna().unique().tolist())
    sel_sig = st.sidebar.selectbox("Clinical Significance", sigs)
    
    if sel_gene != "All":
        df = df[df['gene'] == sel_gene]
    if sel_chr != "All":
        df = df[df['chromosome'].astype(str) == sel_chr]
    if sel_sig != "All":
        df = df[df['clinical_significance'] == sel_sig]
        
    st.markdown(f"**Showing {len(df)} variants**")
    get_csv_download_link(df, "filtered_variants.csv")
    
    tab1, tab2 = st.tabs(["Visualizations", "Data Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            chr_counts = df['chromosome'].value_counts().reset_index()
            chr_counts.columns = ['Chromosome', 'Count']
            render_dynamic_chart(chr_counts, 'Chromosome', 'Count', "Variants by Chromosome", default_type="Bar (Vertical)", key="var_chr", color_sequence=['#2979ff'])
            
        with col2:
            sig_counts = df['clinical_significance'].value_counts().reset_index()
            sig_counts.columns = ['Significance', 'Count']
            render_dynamic_chart(sig_counts, 'Significance', 'Count', "Clinical Significance Distribution", default_type="Pie / Donut", key="var_sig")
            
        # Also top clinical conditions if available
        valid_conds = df[df['condition'] != "Not provided"]
        if not valid_conds.empty:
            cond_counts = valid_conds['condition'].value_counts().head(10).reset_index()
            cond_counts.columns = ['Condition', 'Count']
            render_dynamic_chart(cond_counts, 'Condition', 'Count', "Top Associated Clinical Conditions", default_type="Bar (Horizontal)", key="var_cond", color_sequence=['#d500f9'])
            
    with tab2:
        st.dataframe(df[['variation_id', 'gene', 'condition', 'clinical_significance', 'review_status', 'chromosome', 'position']])


def show_gene_explorer():
    st.title("Gene / Phenotype Explorer")
    
    genes = set()
    if not data['hpo_gene'].empty:
        genes.update(data['hpo_gene']['gene_symbol'].dropna().unique())
    if not data['clinvar'].empty:
        genes.update(data['clinvar']['gene'].dropna().unique())
    if not data['gwas'].empty:
        genes.update(data['gwas']['mapped_gene'].dropna().unique())
        
    if not genes:
        st.warning("No gene data available.")
        return
        
    sel_gene = st.selectbox("Select a Gene", sorted(list(genes)))
    
    hpo_subset = data['hpo_gene'][data['hpo_gene']['gene_symbol'] == sel_gene] if not data['hpo_gene'].empty else pd.DataFrame()
    clinvar_subset = data['clinvar'][data['clinvar']['gene'] == sel_gene] if not data['clinvar'].empty else pd.DataFrame()
    gwas_subset = data['gwas'][data['gwas']['mapped_gene'] == sel_gene] if not data['gwas'].empty else pd.DataFrame()
    
    st.markdown(f"## Gene Summary: {sel_gene}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HPO Phenotypes", len(hpo_subset))
    col2.metric("ClinVar Variants", len(clinvar_subset))
    col3.metric("ClinVar Conditions", clinvar_subset['condition'].nunique() if not clinvar_subset.empty else 0)
    col4.metric("GWAS Associations", len(gwas_subset))
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if not hpo_subset.empty:
            phenos = hpo_subset['hpo_name'].value_counts().head(10).reset_index()
            phenos.columns = ['Phenotype', 'Count']
            render_dynamic_chart(phenos, 'Phenotype', 'Count', "HPO Phenotypes", default_type="Bar (Horizontal)", key="gene_hpo", color_sequence=['#00e5ff'])
            st.dataframe(hpo_subset[['hpo_id', 'hpo_name']].drop_duplicates(), hide_index=True)
        else:
            st.subheader("HPO Phenotypes")
            st.info("No HPO annotations for this gene.")
            
    with c2:
        if not clinvar_subset.empty:
            conds = clinvar_subset['condition'].value_counts().head(10).reset_index()
            conds.columns = ['Condition', 'Count']
            render_dynamic_chart(conds, 'Condition', 'Count', "ClinVar Conditions", default_type="Bar (Horizontal)", key="gene_clin", color_sequence=['#00e676'])
            st.dataframe(clinvar_subset[['variation_id', 'condition', 'clinical_significance']].drop_duplicates().head(50), hide_index=True)
        else:
            st.subheader("ClinVar Conditions")
            st.info("No ClinVar variants for this gene.")
            
    with c3:
        if not gwas_subset.empty:
            traits = gwas_subset['trait'].value_counts().head(10).reset_index()
            traits.columns = ['Trait', 'Count']
            render_dynamic_chart(traits, 'Trait', 'Count', "GWAS Traits", default_type="Bar (Horizontal)", key="gene_gwas", color_sequence=['#ff9100'])
            st.dataframe(gwas_subset[['rs_id', 'trait', 'p_value']].drop_duplicates(), hide_index=True)
        else:
            st.subheader("GWAS Traits")
            st.info("No GWAS associations for this gene.")


def show_network_explorer():
    st.title("Network Explorer")
    if data['rels'].empty:
        st.warning("Relationships not built yet.")
        return
        
    focal_entity = st.text_input("Enter a Gene, Disease, or Variant to focus on", "BRCA1")
    direct_only = st.checkbox("Show direct relationships only", value=True)
    
    if focal_entity:
        df = data['rels']
        
        # Filter for direct
        direct_edges = df[(df['source_name'] == focal_entity) | (df['target_name'] == focal_entity) | 
                          (df['source_id'] == focal_entity) | (df['target_id'] == focal_entity)]
                          
        if direct_edges.empty:
            st.info("No relationships found for this entity.")
            return
            
        nodes = set(direct_edges['source_name']).union(set(direct_edges['target_name']))
        
        if not direct_only:
            # 2nd degree
            second_edges = df[(df['source_name'].isin(nodes)) | (df['target_name'].isin(nodes))]
            direct_edges = pd.concat([direct_edges, second_edges]).drop_duplicates()
            
        # Networkx
        G = nx.from_pandas_edgelist(direct_edges, 'source_name', 'target_name', ['relationship', 'dataset_source'])
        
        st.markdown(f"**Nodes**: {G.number_of_nodes()} | **Edges**: {G.number_of_edges()} | **Connected Components**: {nx.number_connected_components(G)}")
        
        # Plotly graph
        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        node_x = []
        node_y = []
        text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=text, textposition="top center",
                                hoverinfo='text', marker=dict(size=10, color='#00e5ff', line_width=2))
                                
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
                
        st.plotly_chart(fig, use_container_width=True)

def show_disease_explorer():
    st.title("Disease / Phenotype Explorer")
    if data['hpo_dis'].empty:
        st.warning("Missing phenotype.hpoa. Place the file in data/raw/ and run prepare script.")
        return
        
    df = data['hpo_dis']
    diseases = sorted(df['disease_name'].dropna().unique())
    sel_dis = st.selectbox("Select Disease/Condition", diseases)
    
    d_df = df[df['disease_name'] == sel_dis]
    st.markdown(f"### {sel_dis}")
    st.metric("Phenotype Annotations", len(d_df))
    
    col1, col2 = st.columns(2)
    with col1:
        phenos = d_df['hpo_id'].value_counts().head(10).reset_index()
        phenos.columns = ['Phenotype ID', 'Count']
        render_dynamic_chart(phenos, 'Phenotype ID', 'Count', "Top Phenotypes", default_type="Bar (Horizontal)", key="dis_pheno", color_sequence=['#00e5ff'])
    with col2:
        if 'evidence' in d_df.columns:
            ev_counts = d_df['evidence'].value_counts().reset_index()
            ev_counts.columns = ['Evidence', 'Count']
            render_dynamic_chart(ev_counts, 'Evidence', 'Count', "Evidence Distribution", default_type="Pie / Donut", key="dis_ev", color_sequence=['#ffd600'])
            
    st.dataframe(d_df[['database_id', 'hpo_id', 'evidence', 'reference', 'frequency', 'onset']])
    get_csv_download_link(d_df, "disease_phenotypes.csv")

def show_gwas_explorer():
    st.title("GWAS Explorer")
    if data['gwas'].empty:
        st.warning("GWAS cache is missing or data has not yet been prepared. Run the refresh script.")
        return
        
    df = data['gwas']
    st.sidebar.subheader("Filters")
    
    traits = ["All"] + sorted(df['trait'].dropna().unique().tolist())
    sel_trait = st.sidebar.selectbox("Trait", traits)
    
    genes = ["All"] + sorted(df['mapped_gene'].dropna().unique().tolist())
    sel_gene = st.sidebar.selectbox("Gene", genes)
    
    if sel_trait != "All":
        df = df[df['trait'] == sel_trait]
    if sel_gene != "All":
        df = df[df['mapped_gene'] == sel_gene]
        
    st.markdown(f"**Showing {len(df)} associations**")
    get_csv_download_link(df, "gwas_associations.csv")
    
    col1, col2 = st.columns(2)
    with col1:
        tr_counts = df['trait'].value_counts().head(10).reset_index()
        tr_counts.columns = ['Trait', 'Count']
        render_dynamic_chart(tr_counts, 'Trait', 'Count', "Top Traits", default_type="Bar (Horizontal)", key="gwas_tr", color_sequence=['#ff9100'])
    with col2:
        gn_counts = df['mapped_gene'].value_counts().head(10).reset_index()
        gn_counts.columns = ['Gene', 'Count']
        render_dynamic_chart(gn_counts, 'Gene', 'Count', "Top Genes", default_type="Bar (Horizontal)", key="gwas_gn", color_sequence=['#00e676'])
        
    st.subheader("Associations")
    st.dataframe(df[['rs_id', 'trait', 'mapped_gene', 'chromosome', 'position', 'p_value', 'study_accession']])

def show_provenance():
    st.title("Data Sources & Methodology")
    
    st.info("This project is an educational and exploratory data visualization system. The relationships shown are derived from public biomedical databases and represent reported annotations or associations. They do not by themselves establish causation, individual disease risk, or a medical diagnosis.")
    
    st.markdown("""
    ### Human Phenotype Ontology (HPO)
    - **Provider**: Monarch Initiative / HPO Consortium
    - **Purpose**: Connects diseases and genes to standardized phenotype vocabulary.
    - **Methodology**: Direct parsing of `phenotype.hpoa` and `genes_to_phenotype.txt`.
    - **Note**: Phenotype annotations indicate that a characteristic has been reported in patients, not necessarily population prevalence.
    
    ### NCBI ClinVar
    - **Provider**: NCBI
    - **Purpose**: Aggregates reports of genomic variants and their relationship to human health.
    - **Methodology**: Targeted queries of key genes via E-utilities API.
    
    ### NHGRI-EBI GWAS Catalog
    - **Provider**: NHGRI and EBI
    - **Purpose**: Provides a consistent, searchable, visualisable and freely available database of published SNP-trait associations.
    - **Methodology**: Targeted queries via GWAS REST API v2 for associations involving key genes.
    - **Note**: A GWAS association represents a statistical correlation in a studied population, not necessarily a causal effect.
    """)
    
    if DATASET_PROFILE_FILE.exists():
        st.markdown("### Dataset Profiles")
        prof = pd.read_parquet(DATASET_PROFILE_FILE)
        st.dataframe(prof, hide_index=True)

def show_dna_classifier():
    st.title("DNA Sequence Disease Classifier")
    st.markdown("Machine Learning-based diagnostic screening and k-mer motif attribution system for candidate genetic conditions.")
    
    # Pre-defined sample sequences for quick testing
    sample_sequences = {
        "CFTR (Cystic Fibrosis)": "GTAGTAGGTCTTTGGCATTAGGAGCTTGAGCCCAGACGGCCCTAGCAGGGACCCCAGCGCCCGAGAGACCATGCAGAGGTCGCCTCTGGAAAAGGCCAGCGTTGTCTCCAAACTTTTTTT",
        "BRCA1 (Breast/Ovarian Cancer)": "AATGAATGTAGAAAAGGCTGAATTCTGTAATAAAAGCAAACAGCCTGGCTTAGCAAGGAGCCAACATAACAGATGGGCTGGAAGTAAGGAAACATGTAATGATAGGCGGACTCCCAGCAC",
        "HBB (Beta-Thalassemia)": "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAG",
        "LDLR (Hypercholesterolemia)": "GTGCAATCGCGGGAAGCCAGGGTTTCCAGCTAGGACACAGCAGGTCGTGATCCGGGTCGGGACACTGCCTGGCAGAGGCTGCGAGCATGGGGCCCTGGGGCTGGAAATTGCGCTGGACCG",
        "TP53 (Li-Fraumeni Syndrome)": "CTCAAAAGTCTAGAGCCACCGTCCAGGGAGCAGGTAGCTGCTGGGCTCCGGGGACACTTTGCGTTCGGGCTGGGAGCGTGCTTTCCACGACGGTGACACGCTTCCCTGGATTGGCAGCCA",
        "Invalid Sequence (Error Demo)": "ATGCXGAT19ZATCGAATCGATC!#999INVALID"
    }
    
    # Initialize session state for sequence
    if "user_dna_sequence" not in st.session_state:
        st.session_state["user_dna_sequence"] = sample_sequences["CFTR (Cystic Fibrosis)"]
        
    st.subheader("1. Input DNA Sequence")
    st.caption("Paste a FASTA format or raw nucleotide sequence (A, C, G, T), or click a pre-loaded sample below:")
    
    # Demo buttons
    bcols = st.columns(len(sample_sequences))
    for col, (label, seq_val) in zip(bcols, sample_sequences.items()):
        if col.button(label, use_container_width=True):
            st.session_state["user_dna_sequence"] = seq_val
            st.rerun()
            
    input_seq = st.text_area(
        "DNA Nucleotide Sequence", 
        value=st.session_state.get("user_dna_sequence", ""), 
        height=140,
        placeholder="Enter nucleotide sequence (A, C, G, T)..."
    )
    
    classify_clicked = st.button("🧬 Verify & Classify Sequence", type="primary")
    
    # If there is an input sequence, perform verification and classification
    if input_seq:
        st.markdown("---")
        st.subheader("2. Biological Sequence Verification")
        
        verification = verify_dna_sequence(input_seq)
        
        if not verification["is_valid"]:
            st.error(f"❌ **Validation Failed**: {verification['error_message']}")
            st.warning("⚠️ Biological Requirement: Real DNA sequences only contain nucleotides Adenine (A), Cytosine (C), Guanine (G), and Thymine (T). Please correct illegal characters or ensure sequence is at least 20 bp.")
            return
            
        st.success("✅ **Sequence Validation Passed**: 100% biological nucleotide compliance verified.")
        
        # Display sequence statistics metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sequence Length", f"{verification['length']} bp")
        m2.metric("GC Content", f"{verification['gc_content']}%")
        m3.metric("AT Content", f"{verification['at_content']}%")
        m4.metric("Purine/Pyrimidine Ratio", f"{verification['purine_pyrimidine_ratio']}")
        
        # Nucleotide composition mini chart
        pcts = verification["nucleotide_percentages"]
        comp_df = pd.DataFrame([
            {"Nucleotide": n, "Percentage (%)": pcts[n], "Count": verification["nucleotide_counts"].get(n, 0)}
            for n in ["A", "C", "G", "T"]
        ])
        render_dynamic_chart(comp_df, "Nucleotide", "Percentage (%)", "Nucleotide Composition Distribution", default_type="Bar (Vertical)", key="ml_comp", color_sequence=['#2979ff', '#00e5ff', '#00e676', '#ff9100'])
        
        st.markdown("---")
        st.subheader("3. Disease Classification Results")
        
        # Inference
        with st.spinner("Classifying sequence and attributing k-mer features..."):
            prediction = predict_and_justify(verification["cleaned_sequence"])
            
        pred_disease = prediction["predicted_disease"]
        conf_pct = prediction["confidence_pct"]
        conf_level = prediction["confidence_level"]
        bio = prediction["biological_context"]
        
        # Prominent Result Banner
        st.markdown(f"""
        <div style="background: rgba(0, 229, 255, 0.1); border-left: 5px solid #00e5ff; padding: 1.2rem; border-radius: 6px; margin-bottom: 1.5rem;">
            <h3 style="margin: 0; color: #00e5ff;">Predicted Condition: {pred_disease}</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #f5f5f5;">
                Confidence: <strong style="color: #00e676;">{conf_pct}% ({conf_level})</strong> | 
                Locus: <strong>{', '.join(bio['chromosomes'])}</strong> | 
                Primary Gene(s): <strong>{', '.join(bio['associated_genes'])}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Candidate Diseases Probability Chart
        c_left, c_right = st.columns([3, 2])
        with c_left:
            prob_items = sorted(prediction["all_probabilities"].items(), key=lambda x: x[1])
            prob_df = pd.DataFrame([
                {"Disease Class": k, "Probability (%)": round(v * 100, 1)}
                for k, v in prob_items
            ])
            render_dynamic_chart(prob_df, "Disease Class", "Probability (%)", "Candidate Diseases Probability Distribution", default_type="Bar (Horizontal)", key="ml_prob", color_sequence=['#2979ff'])
            
        with c_right:
            st.markdown("#### Candidate Conditions Table")
            cand_df = pd.DataFrame(prediction["candidate_diseases"])
            cand_df.columns = ["Disease Class", "Probability", "Confidence (%)"]
            st.dataframe(cand_df, hide_index=True, use_container_width=True)
            st.caption("Conditions with calibrated probability >= 5.0% are retained as differential diagnoses.")
            
        st.markdown("---")
        st.subheader("4. Justification for Classified Class")
        
        j_col1, j_col2 = st.columns(2)
        
        with j_col1:
            st.markdown("#### Mathematical Justification (Driving K-mer Motifs)")
            st.markdown(
                "Local feature attribution identified the following distinctive nucleotide substrings (k-mers) "
                "within this sequence that provided the highest positive attribution score for the prediction:"
            )
            motifs_df = pd.DataFrame(prediction["driving_motifs"])
            motifs_df.columns = ["Rank", "K-mer Motif", "Length (bp)", "Attribution Weight"]
            render_dynamic_chart(motifs_df, "K-mer Motif", "Attribution Weight", "Driving K-mer Motifs", default_type="Bar (Horizontal)", key="ml_motifs", color_sequence=['#00e676'])
            st.dataframe(motifs_df, hide_index=True, use_container_width=True)
            
        with j_col2:
            st.markdown("#### Biological & Phenotypic Grounding")
            st.markdown(f"**Associated Gene(s)**: `{', '.join(bio['associated_genes'])}`")
            st.markdown(f"**Chromosomal Locus**: `{', '.join(bio['chromosomes'])}`")
            st.markdown(f"**Inheritance Mode**: `{bio['inheritance']}`")
            st.markdown("**Reported Clinical Phenotypes (from HPO)**:")
            for hp in bio["hpo_phenotypes"]:
                st.markdown(f"- `{hp}`")
            st.markdown(f"**Pathogenic Molecular Mechanism**:\n{bio['mechanism']}")
            
        st.info(f"**Synthesis Justification**: {prediction['justification_summary']}")
        
    st.markdown("---")
    
    # Model Performance & Metrics Justification Section
    st.subheader("5. Model Performance & Metrics Justification")
    
    model_obj, metrics_data = train_and_evaluate_model()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test Accuracy", f"{round(metrics_data['accuracy'] * 100, 2)}%")
    k2.metric("Macro F1-Score", f"{round(metrics_data['f1_macro'] * 100, 2)}%")
    k3.metric("Weighted F1-Score", f"{round(metrics_data['f1_weighted'] * 100, 2)}%")
    k4.metric("5-Fold CV Accuracy", f"{round(metrics_data['cv_accuracy_mean'] * 100, 2)}% ± {round(metrics_data['cv_accuracy_std'] * 100, 2)}%")
    
    tabs = st.tabs(["📊 Confusion Matrix & Classification Report", "🎯 Metrics Justification", "🔒 Data Leakage Prevention Guarantee"])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### Confusion Matrix (Test Set, N = 317)")
            classes_list = metrics_data["classes"]
            cm_matrix = metrics_data["confusion_matrix"]
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=classes_list,
                y=classes_list,
                color_continuous_scale="Blues",
                text_auto=True,
                template="plotly_dark"
            )
            fig_cm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c2:
            st.markdown("##### Per-Class Performance Breakdown")
            rep = metrics_data["classification_report"]
            rows = []
            for cls_name in classes_list:
                if cls_name in rep:
                    rows.append({
                        "Disease Class": cls_name,
                        "Precision": round(rep[cls_name]["precision"], 3),
                        "Recall": round(rep[cls_name]["recall"], 3),
                        "F1-Score": round(rep[cls_name]["f1-score"], 3),
                        "Support": int(rep[cls_name]["support"])
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(f"Evaluated on {metrics_data['test_samples']} independent sequence test samples strictly held out from training.")
            
    with tabs[1]:
        st.markdown("##### Scientific Justification for Selected Classification Metrics")
        just = metrics_data["metric_justifications"]
        st.markdown(f"**1. Accuracy**: {just['accuracy']}")
        st.markdown(f"**2. Precision (Positive Predictive Value)**: {just['precision']}")
        st.markdown(f"**3. Recall (Sensitivity)**: {just['recall']}")
        st.markdown(f"**4. F1-Score (Harmonic Mean)**: {just['f1_score']}")
        st.info("In clinical genomics, optimizing for F1-Score ensures the model avoids biased predictions towards genes with longer transcripts (e.g. APOB or BRCA2) while maintaining sensitivity for short critical loci like HBB.")
        
    with tabs[2]:
        st.markdown("##### Rigorous Data Leakage Prevention Architecture")
        st.markdown("""
        To ensure scientific integrity and eliminate any possibility of data leakage:
        
        1. **Stratified Train/Test Partitioning**: Sequences are split using `train_test_split(..., stratify=y, test_size=0.25, random_state=42)` *strictly prior* to feature extraction.
        2. **Pipeline Encapsulation**: The character k-mer vectorizer (`TfidfVectorizer(ngram_range=(3, 6))`) and the classifier (`LogisticRegression`) are bundled inside a unified `sklearn.pipeline.Pipeline`.
        3. **Isolated Vocabulary & IDF Fitting**: All n-gram vocabulary indices, document frequency thresholds, and Inverse Document Frequencies (IDF) are fitted exclusively on the `X_train` partition. Zero vocabulary or token statistics from `X_test` are observed during training.
        4. **Zero Test Contamination**: The test fold is transformed and evaluated only after the entire pipeline is frozen.
        5. **5-Fold Cross Validation**: Cross-validation is executed strictly within `X_train` using `StratifiedKFold`, ensuring unbiased hyperparameter selection.
        """)

def show_chart_studio():
    st.title("Interactive Graph & Chart Studio")
    st.markdown("Universal visualization workspace: Select any dataset, attributes, aggregation, and plot type on demand.")
    
    dataset_options = {
        "ClinVar Variants (Curated Mutations & Clinical Significance)": ("clinvar", data.get('clinvar', pd.DataFrame())),
        "GWAS Catalog Associations (Variant-Trait-Gene Mappings)": ("gwas", data.get('gwas', pd.DataFrame())),
        "HPO Disease-Phenotype Annotations (Clinical Signs)": ("hpo_dis", data.get('hpo_dis', pd.DataFrame())),
        "HPO Gene-Phenotype Annotations (Gene-Symptom Mappings)": ("hpo_gene", data.get('hpo_gene', pd.DataFrame())),
        "Integrated Knowledge Relationships (Graph Edges)": ("rels", data.get('rels', pd.DataFrame())),
        "ML Reference Sequences (Transcript Fragments & Diseases)": ("ml_seq", data.get('ml_seq', pd.DataFrame()))
    }
    
    selected_dataset_label = st.selectbox("1. Choose Biomedical Dataset to Analyze", list(dataset_options.keys()))
    dataset_key, df_raw = dataset_options[selected_dataset_label]
    
    if df_raw.empty:
        st.warning("Selected dataset is currently empty. Run the data preparation script to populate it.")
        return
        
    st.markdown("---")
    
    # Configuration columns
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    
    with col_cfg1:
        st.markdown("##### 2. Choose Chart Type")
        all_chart_types = [
            "Vertical Bar Chart",
            "Horizontal Bar Chart",
            "Pie / Donut Chart",
            "Line / Trend Chart",
            "Area Chart",
            "Scatter Plot",
            "Treemap",
            "Box Plot",
            "Histogram / Distribution",
            "Funnel Chart"
        ]
        selected_chart_type = st.selectbox("Chart Type", all_chart_types)
        
    with col_cfg2:
        st.markdown("##### 3. Axis / Dimension Configuration")
        available_cols = list(df_raw.columns)
        
        # Categorical / Dimension column
        default_x = "gene" if "gene" in available_cols else ("mapped_gene" if "mapped_gene" in available_cols else available_cols[0])
        x_dim = st.selectbox("Category / X-Axis Column", available_cols, index=available_cols.index(default_x))
        
        # Aggregation or Y-axis
        metric_choices = ["Record Count (Frequency)"] + [c for c in available_cols if c != x_dim]
        y_dim = st.selectbox("Metric / Y-Axis Column", metric_choices)
        
    with col_cfg3:
        st.markdown("##### 4. Filtering & Aesthetics")
        top_n = st.selectbox("Top N Categories", [5, 10, 15, 20, 50, "All"], index=1)
        color_theme = st.selectbox("Color Theme", ["Cyberpunk Cyan", "Plotly Dark", "Viridis", "Turbo", "Plasma", "Emerald Green", "Sunset"])
        
        # Optional breakdown / color by
        color_by_options = ["None"] + [c for c in available_cols if c != x_dim]
        color_by = st.selectbox("Color / Group By (Optional)", color_by_options)
        
    # Process and aggregate data
    with st.spinner("Generating custom visualization..."):
        df_plot = df_raw.copy()
        
        # Color palettes
        theme_map = {
            "Cyberpunk Cyan": ['#00e5ff', '#2979ff', '#00e676', '#ff9100', '#d500f9', '#ffd600', '#ff1744'],
            "Plotly Dark": px.colors.qualitative.Plotly,
            "Viridis": px.colors.sequential.Viridis,
            "Turbo": px.colors.sequential.Turbo,
            "Plasma": px.colors.sequential.Plasma,
            "Emerald Green": ['#00e676', '#00b0ff', '#1de9b6', '#76ff03', '#64ffda', '#00e5ff'],
            "Sunset": px.colors.sequential.Sunset
        }
        active_palette = theme_map.get(color_theme, ['#00e5ff', '#2979ff'])
        
        # Prepare aggregated data if Count is requested
        if y_dim == "Record Count (Frequency)":
            if color_by != "None":
                counts = df_plot.groupby([x_dim, color_by]).size().reset_index(name="Count")
                if top_n != "All":
                    top_x = df_plot[x_dim].value_counts().head(top_n).index
                    counts = counts[counts[x_dim].isin(top_x)]
                df_chart = counts
                metric_col = "Count"
            else:
                counts = df_plot[x_dim].value_counts().reset_index()
                counts.columns = [x_dim, 'Count']
                if top_n != "All":
                    counts = counts.head(top_n)
                df_chart = counts
                metric_col = "Count"
        else:
            # If Y is a specific column
            df_chart = df_plot.dropna(subset=[x_dim, y_dim])
            if top_n != "All":
                top_x = df_chart[x_dim].value_counts().head(top_n).index
                df_chart = df_chart[df_chart[x_dim].isin(top_x)]
            metric_col = y_dim
            
        fig = None
        color_arg = color_by if (color_by != "None" and color_by in df_chart.columns) else None
        
        if selected_chart_type == "Vertical Bar Chart":
            fig = px.bar(df_chart, x=x_dim, y=metric_col, color=color_arg, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Horizontal Bar Chart":
            fig = px.bar(df_chart, x=metric_col, y=x_dim, color=color_arg, orientation='h', template="plotly_dark", color_discrete_sequence=active_palette)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
        elif selected_chart_type == "Pie / Donut Chart":
            fig = px.pie(df_chart, names=x_dim, values=metric_col, template="plotly_dark", hole=0.4, color_discrete_sequence=active_palette)
        elif selected_chart_type == "Line / Trend Chart":
            fig = px.line(df_chart, x=x_dim, y=metric_col, color=color_arg, markers=True, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Area Chart":
            fig = px.area(df_chart, x=x_dim, y=metric_col, color=color_arg, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Scatter Plot":
            fig = px.scatter(df_chart, x=x_dim, y=metric_col, color=color_arg, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Treemap":
            path_cols = [x_dim] if color_arg is None else [color_arg, x_dim]
            fig = px.treemap(df_chart, path=path_cols, values=metric_col, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Box Plot":
            fig = px.box(df_chart, x=x_dim, y=metric_col, color=color_arg, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Histogram / Distribution":
            fig = px.histogram(df_chart, x=x_dim, y=None if y_dim == "Record Count (Frequency)" else metric_col, color=color_arg, template="plotly_dark", color_discrete_sequence=active_palette)
        elif selected_chart_type == "Funnel Chart":
            fig = px.funnel(df_chart, x=metric_col, y=x_dim, color=color_arg, template="plotly_dark", color_discrete_sequence=active_palette)
            
        if fig:
            fig.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Data table & download
        with st.expander("📄 View Plotted Data & Export CSV", expanded=False):
            st.dataframe(df_chart, hide_index=True, use_container_width=True)
            get_csv_download_link(df_chart, f"custom_plot_{x_dim}.csv")

# Router
if selection == "Overview":
    show_overview()
elif selection == "Variant Explorer":
    show_variant_explorer()
elif selection == "Gene / Phenotype Explorer":
    show_gene_explorer()
elif selection == "Disease / Phenotype Explorer":
    show_disease_explorer()
elif selection == "GWAS Explorer":
    show_gwas_explorer()
elif selection == "Network Explorer":
    show_network_explorer()
elif selection == "Interactive Chart Studio":
    show_chart_studio()
elif selection == "Data Sources & Methodology":
    show_provenance()
elif selection == "DNA Sequence Disease Classifier":
    show_dna_classifier()
    
# Footer
st.markdown("<div class='footer'>Educational data visualization. Not medical advice.</div>", unsafe_allow_html=True)
