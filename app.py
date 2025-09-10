import os
import json
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import xgboost, sklearn

# =========================
# ÉTAT SESSION (confirmations utilisateur)
# =========================
if 'temp_confirmed_types' not in st.session_state:
    st.session_state.temp_confirmed_types = {}
if 'confirmed_types' not in st.session_state:
    st.session_state.confirmed_types = {}

# =========================
# REGEX (IDENTIQUES AU TRAINING)
# =========================
RE_NUMERIC = re.compile(r'NUMBER\(\d+(,\d+)?\)|INTEGER|DECIMAL|FLOAT|NUMERIC|BIGINT', re.I)
RE_TEXT    = re.compile(r'CHAR\(\d+\)|VARCHAR2\(\d+\)|VARCHAR|TEXT|STRING', re.I)
RE_DATE    = re.compile(r'DATE|TIMESTAMP|TIME', re.I)
RE_KEYS    = re.compile(r'ID|CODE|NUM|KEY|REF|LIBL', re.I)
RE_KEYS_RELATION = re.compile(r'CODE|TYPE|CODE TYPE|CODE ETAT', re.I)  # on ne touche pas à ta regex
RE_MEAS    = re.compile(r'MONTANT|AMOUNT|VALEUR|VALUE|PRICE|PRIX|QTE|QTY|COUNT|TOTAL|SUM|MT_|SOLD_|VALR_', re.I)
RE_CODE_PREFIX = re.compile(r'^CODE_|^TYPE_', re.I)
RE_FACT_PREFIX = re.compile(r'FACT|MESURE|FLUX|TRANSACTION|EVNM|MVT_|ECRT_|BALN_', re.I)

# =========================
# HELPERS de normalisation (pour fiabiliser les matches)
# =========================
def _norm(s: str) -> str:
    # Remplace les NBSP, trim, passe en MAJ
    return str(s).replace('\u00A0', ' ').strip().upper()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'LIBELLE_DU_SEGMENT' in df.columns:
        df['LIBELLE_DU_SEGMENT'] = df['LIBELLE_DU_SEGMENT'].map(_norm)
    if 'NOM_EPURE_DE_LA_RUBRIQUE' in df.columns:
        df['NOM_EPURE_DE_LA_RUBRIQUE'] = df['NOM_EPURE_DE_LA_RUBRIQUE'].map(_norm)
    if 'NOM_DE_LA_COLONNE_NORME_ADD' in df.columns:
        df['NOM_DE_LA_COLONNE_NORME_ADD'] = df['NOM_DE_LA_COLONNE_NORME_ADD'].map(_norm)
    # Compat noms courants
    for alt in ['Nom', 'Code']:
        if alt in df.columns:
            df[alt] = df[alt].map(_norm)
    return df

# =========================
# CHARGEMENT MODÈLES / FEATURES
# =========================
@st.cache_resource
def load_xgb_model():
    xgb_model = XGBClassifier()
    xgb_model.load_model('best_xgb_model.json')
    return xgb_model

xgb_model = load_xgb_model()

@st.cache_resource
def load_model_features():
    """
    Source de vérité : best_xgb_features.json sauvegardé à l'entraînement.
    Fallbacks: training_features_by_table.csv puis feature_names_in_ si dispo.
    """
    if os.path.exists('best_xgb_features.json'):
        with open('best_xgb_features.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    if os.path.exists('training_features_by_table.csv'):
        base = pd.read_csv('training_features_by_table.csv')
        cols = [c for c in base.columns if c not in ('Table_Name', 'Table_Type')]
        return cols

    if hasattr(xgb_model, 'feature_names_in_'):
        return list(xgb_model.feature_names_in_)

    raise RuntimeError(
        "Impossible de retrouver les features du modèle. "
        "Assure-toi d'avoir 'best_xgb_features.json' ou 'training_features_by_table.csv'."
    )

# =========================
# UTILS : RELATIONS FACT-DIM
# =========================
def detect_fact_dim_links(original_df, type_predictions_df, prob_diff_threshold=0.2):
   
    df = normalize_df(original_df)
    type_map = dict(zip(
        type_predictions_df['Table_Name'].map(_norm),
        type_predictions_df['Type Prédit']
    ))
    candidate_name_cols = []
    for c in ['NOM_EPURE_DE_LA_RUBRIQUE', 'NOM_DE_LA_COLONNE_NORME_ADD', 'Nom', 'Code']:
        if c in df.columns:
            candidate_name_cols.append(c)
    if not candidate_name_cols:
        candidate_name_cols = ['NOM_EPURE_DE_LA_RUBRIQUE']
    all_tables = df['LIBELLE_DU_SEGMENT'].unique()

    def cols_for_table(t):
        subset = df[df['LIBELLE_DU_SEGMENT'] == t]
        s = set()
        for c in candidate_name_cols:
            s |= set(subset[c].dropna())
        return s

    table_columns = {t: cols_for_table(t) for t in all_tables}
    fact_tables = [t for t in all_tables if type_map.get(_norm(t)) == 'FAIT']
    dim_tables  = [t for t in all_tables if type_map.get(_norm(t)) == 'DIMENSION']

    relations = []
    for fact in fact_tables:
        fact_cols = table_columns.get(fact, set())
        for dim in dim_tables:
            dim_cols = table_columns.get(dim, set())
            common_keys = [col for col in (fact_cols & dim_cols) if RE_KEYS_RELATION.search(col)]
            if common_keys:
                relations.append({
                    'Table_Fact': fact,
                    'Table_Dimension': dim,
                    'Colonnes_Communes': ', '.join(sorted(common_keys)),
                    'Nb_Colonnes_Communes': len(common_keys)
                })
    return pd.DataFrame(relations)

# =========================
# FEATURES LOCALES (mêmes règles que training)
# =========================
def has_code_prefix(table_name):
    return 1 if RE_CODE_PREFIX.search(str(table_name)) else 0

def has_measure_prefix(table_name):
    return 1 if RE_FACT_PREFIX.search(str(table_name)) else 0

def count_key_like_columns(column_names):
    return sum(1 for c in column_names if RE_KEYS.search(str(c)))

def count_measure_like_columns(column_names):
    return sum(1 for c in column_names if RE_MEAS.search(str(c)))

# =========================
# PRÉTRAITEMENT EXCEL (aligné sur training + normalisation)
# =========================
def preprocess_excel_file(file):
    """
    Construit les features EXACTEMENT comme au training.
    Seules colonnes nécessaires : LIBELLE_DU_SEGMENT, NOM_EPURE_DE_LA_RUBRIQUE, TYPE_DONNEES_COLONNE, PK.
    'Table_Type' est optionnelle (pour calculer la précision si présente).
    """
    try:
        df = pd.read_excel(file)
        df = normalize_df(df)

        required_columns = [
            'LIBELLE_DU_SEGMENT',
            'NOM_EPURE_DE_LA_RUBRIQUE',
            'TYPE_DONNEES_COLONNE',
            'PK'
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            st.error(f"Colonnes manquantes: {missing}")
            return None, None

        agg_dict = {
            'NOM_EPURE_DE_LA_RUBRIQUE': ['count', lambda x: [str(v).upper() for v in x.fillna('')]],
            'TYPE_DONNEES_COLONNE': lambda x: (
                sum(1 for t in x if RE_NUMERIC.search(str(t))),
                sum(1 for t in x if RE_TEXT.search(str(t))),
                sum(1 for t in x if RE_DATE.search(str(t)))
            ),
            'PK': lambda x: sum(1 for p in x if str(p).strip().upper() == 'O')
        }

        has_table_type = 'Table_Type' in df.columns
        if has_table_type:
            agg_dict['Table_Type'] = 'first'

        grouped = df.groupby('LIBELLE_DU_SEGMENT').agg(agg_dict).reset_index()

        expected_cols = ['Table_Name', 'Num_Columns', 'Column_Names', 'Type_Counts', 'Num_PKs']
        if has_table_type:
            expected_cols.append('Table_Type')
        grouped.columns = expected_cols

        grouped['Num_Columns'] = grouped['Num_Columns'].fillna(0).replace(0, 1)
        grouped['Num_Numeric'] = grouped['Type_Counts'].apply(lambda x: x[0]).fillna(0)
        grouped['Num_Text']    = grouped['Type_Counts'].apply(lambda x: x[1]).fillna(0)
        grouped['Num_Date']    = grouped['Type_Counts'].apply(lambda x: x[2]).fillna(0)
        grouped['Num_PKs']     = grouped['Num_PKs'].fillna(0).astype(int)

        grouped['Pct_Numeric'] = grouped['Num_Numeric'] / grouped['Num_Columns']
        grouped['Pct_Text']    = grouped['Num_Text']    / grouped['Num_Columns']
        grouped['Pct_Date']    = grouped['Num_Date']    / grouped['Num_Columns']
        grouped['PK_Ratio']    = grouped['Num_PKs']     / grouped['Num_Columns']

        grouped['Has_Code_Prefix']    = grouped['Table_Name'].apply(has_code_prefix)
        grouped['Has_Measure_Prefix'] = grouped['Table_Name'].apply(has_measure_prefix)

        grouped['Num_Key_Like']     = grouped['Column_Names'].apply(count_key_like_columns)
        grouped['Num_Measure_Like'] = grouped['Column_Names'].apply(count_measure_like_columns)

        grouped = grouped.drop(columns=['Type_Counts'])
        grouped = grouped.fillna(0)

        # On renvoie aussi df original (utilisé pour détecter les relations)
        return grouped, df

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier Excel : {str(e)}")
        return None, None

# =========================
# UI (inchangé dans la forme/affichage)
# =========================
st.title("Prédiction de type de table (FACT ou DIMENSION)")
st.header("Tester avec un fichier Excel")

prob_diff_threshold = st.slider(
    "Seuil de différence de probabilité pour considérer une table comme incertaine",
    min_value=0.0,
    max_value=0.5,
    value=0.15,
    step=0.05,
    help="Si |P(FACT) - P(DIM)| < seuil, la table est considérée comme incertaine."
)

uploaded_file = st.file_uploader("Charger un fichier Excel (.xlsx)", type="xlsx")

if uploaded_file:
    grouped, original_df = preprocess_excel_file(uploaded_file)
    if grouped is not None:
        st.write("Caractéristiques calculées :", grouped)

        if grouped['Table_Name'].duplicated().any():
            st.warning("Attention : Certains noms de tables sont dupliqués.")

        model_features = [c for c in load_model_features() if c != 'Table_Type']

        X_test = grouped.copy()
        for col in ['Table_Name', 'Table_Type', 'Column_Names']:
            if col in X_test.columns:
                X_test = X_test.drop(columns=col)

        for col in model_features:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[model_features].astype('float32')

        predictions = xgb_model.predict(X_test)
        predictions_proba = xgb_model.predict_proba(X_test)

        results = pd.DataFrame({
            'Table_Name': grouped['Table_Name'],
            'Probabilité FACT': predictions_proba[:, 1],
            'Probabilité DIMENSION': predictions_proba[:, 0],
            'Type Prédit': np.where(predictions == 1, 'FAIT', 'DIMENSION'),
            'Confiance': np.maximum(predictions_proba[:, 1], predictions_proba[:, 0]),
            'Confirmed': False
        })

        for table, typ in st.session_state.confirmed_types.items():
            mask = results['Table_Name'] == table
            if mask.any():
                results.loc[mask, 'Type Prédit'] = typ
                results.loc[mask, 'Confirmed'] = True

        if 'Table_Type' in grouped.columns:
            results['Vrai Type'] = grouped['Table_Type'].map({1: 'FAIT', 0: 'DIMENSION'})

        uncertain_tables = results[
            (np.abs(results['Probabilité FACT'] - results['Probabilité DIMENSION']) < prob_diff_threshold)
            & (~results['Confirmed'])
        ]['Table_Name'].tolist()

        st.subheader("Résultats des prédictions")
        st.dataframe(results.drop(columns=['Confirmed']))

        if uncertain_tables:
            st.warning(
                f" {len(uncertain_tables)} table(s) ont des probabilités proches "
                f"(différence < {prob_diff_threshold}). Confirme le type :"
            )
            uncertain_data = []
            for table in uncertain_tables:
                table_mask = grouped['Table_Name'] == table
                if not table_mask.any():
                    continue
                columns = grouped.loc[table_mask, 'Column_Names'].values[0]
                result_mask = results['Table_Name'] == table
                if not result_mask.any():
                    continue

                prob_fact = results.loc[result_mask, 'Probabilité FACT'].values[0]
                prob_dim  = results.loc[result_mask, 'Probabilité DIMENSION'].values[0]
                default_type = st.session_state.temp_confirmed_types.get(
                    table, results.loc[result_mask, 'Type Prédit'].values[0]
                )

                confirmed_type = st.radio(
                    f"Type pour {table}",
                    ['DIMENSION', 'FAIT'],
                    index=['DIMENSION', 'FAIT'].index(default_type) if default_type in ['DIMENSION', 'FAIT'] else 0,
                    key=f"radio_{table}",
                    horizontal=True
                )
                st.session_state.temp_confirmed_types[table] = confirmed_type

                uncertain_data.append({
                    'Table': table,
                    'Probabilité FACT': f"{prob_fact:.3f}",
                    'Probabilité DIMENSION': f"{prob_dim:.3f}",
                    'Colonnes clés': ', '.join(columns),
                    'Type choisi': confirmed_type
                })

            if uncertain_data:
                df_uncertain = pd.DataFrame(uncertain_data)

                def highlight_type(val):
                    return 'background-color: lightgreen' if val == 'FAIT' else 'background-color: lightblue'

                st.dataframe(
                    df_uncertain.style.applymap(highlight_type, subset=['Type choisi'])
                )

            if st.button("Valider les modifications"):
                st.session_state.confirmed_types.update(st.session_state.temp_confirmed_types)
                st.session_state.temp_confirmed_types = {}
                st.success("Modifications validées !")
                st.rerun()
        else:
            st.info("Aucune table incertaine détectée (différence de probabilité >= seuil ou toutes confirmées).")

        if 'Table_Type' in grouped.columns and not grouped['Table_Type'].isna().all():
            accuracy = accuracy_score(
                grouped['Table_Type'].map({1: 1, 0: 0}),
                results['Type Prédit'].map({'FAIT': 1, 'DIMENSION': 0})
            )
            st.write(f"**Précision sur le fichier chargé :** {accuracy:.4f}")

        relations_df = detect_fact_dim_links(original_df, results, prob_diff_threshold=prob_diff_threshold)
        st.subheader("Relations détectées entre tables de faits et dimensions")
        if relations_df.empty:
            st.info("Aucune relation détectée.")
        else:
            st.dataframe(relations_df)

        modifications_data = []
        for table, typ in st.session_state.confirmed_types.items():
            result_mask = results['Table_Name'] == table
            if not result_mask.any():
                continue
            grouped_mask = grouped['Table_Name'] == table
            if not grouped_mask.any():
                continue
            idx = grouped[grouped_mask].index[0]
            original_prediction = 'FAIT' if predictions[idx] == 1 else 'DIMENSION'
            if typ != original_prediction:
                modifications_data.append({
                    'Table_Name': table,
                    'Type Initial': original_prediction,
                    'Type Corrigé': typ
                })
        modifications_df = pd.DataFrame(modifications_data)

        # On relit le fichier d'entrée "brut" pour garder EXACTEMENT les colonnes et l'ordre d'origine
        try:
            uploaded_file.seek(0)  # revenir au début du buffer
            input_raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            input_raw_df = None
            st.warning(f"Impossible de relire le fichier d'entrée pour la feuille 'Predictions' : {e}")

        if input_raw_df is not None:
            # Mapping table -> type (après corrections éventuelles)
            type_map = dict(zip(results['Table_Name'].map(_norm), results['Type Prédit']))

            # Clé de jointure basée sur LIBELLE_DU_SEGMENT (normalisée côté clé, pas côté data)
            if 'LIBELLE_DU_SEGMENT' not in input_raw_df.columns:
                st.error("La colonne 'LIBELLE_DU_SEGMENT' est absente du fichier d'entrée : impossible d'ajouter la colonne 'TYPE_DIMENSIONNEL_TABLE'.")
                predictions_sheet_df = results.copy()  # fallback
            else:
                tmp_df = input_raw_df.copy()
                tmp_df['_TABLE_KEY_'] = tmp_df['LIBELLE_DU_SEGMENT'].astype(str).map(_norm)

                # >>> Création de la colonne demandée
                tmp_df['TYPE_DIMENSIONNEL_TABLE'] = tmp_df['_TABLE_KEY_'].map(type_map).fillna('UNKNOWN')
                tmp_df.drop(columns=['_TABLE_KEY_'], inplace=True)

                # Renommer TYPE_DONNEES_COLONNE -> FORMAT uniquement pour le fichier exporté
                if 'TYPE_DONNEES_COLONNE' in tmp_df.columns:
                    tmp_df.rename(columns={'TYPE_DONNEES_COLONNE': 'FORMAT'}, inplace=True)

                # Mettre TYPE_DIMENSIONNEL_TABLE en dernière colonne
                cols = [c for c in tmp_df.columns if c != 'TYPE_DIMENSIONNEL_TABLE'] + ['TYPE_DIMENSIONNEL_TABLE']
                predictions_sheet_df = tmp_df.loc[:, cols]
        else:
            predictions_sheet_df = results.copy()  # fallback si relire échoue

        # Export Excel multi-onglets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Feuille 'Predictions' = fichier d'entrée + TYPE_DIMENSIONNEL_TABLE (et FORMAT renommé)
            predictions_sheet_df.to_excel(writer, index=False, sheet_name='Predictions')

            relations_df.to_excel(writer, index=False, sheet_name='Relations_FACT_DIM')
            if not modifications_df.empty:
                modifications_df.to_excel(writer, index=False, sheet_name='Modifications_Manuelles')
        output.seek(0)
        st.download_button(
            label="Télécharger les prédictions et relations (Excel)",
            data=output,
            file_name="predictions_et_relations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================
# SIDEBAR DEBUG (inchangé)
# =========================
if st.sidebar.button("Afficher les features du modèle"):
    st.sidebar.write("**Features attendues par le modèle :**")
    st.sidebar.write(load_model_features())

if st.sidebar.checkbox("Mode debug features (comparaison training)"):
    try:
        base = pd.read_csv('training_features_by_table.csv')
        if uploaded_file and 'grouped' in locals() and grouped is not None:
            sel = st.sidebar.selectbox("Table à comparer", grouped['Table_Name'])
            g = grouped[grouped['Table_Name'] == sel].drop(columns=['Column_Names', 'Table_Type'], errors='ignore').set_index('Table_Name')
            b = base[base['Table_Name'] == sel].set_index('Table_Name')
            if b.empty:
                st.sidebar.warning("Table absente du CSV d'entraînement.")
            else:
                common = [c for c in g.columns if c in b.columns and c != 'Table_Type']
                diff = (g[common].astype(float).round(8) - b[common].astype(float).round(8)).T
                st.sidebar.write("Différences (app - train) ≠ 0 ⇒ drift :")
                st.sidebar.dataframe(diff[diff.ne(0).any(axis=1)])
        else:
            st.sidebar.info("Charge un fichier Excel pour activer la comparaison.")
    except Exception as e:
        st.sidebar.error(f"Debug impossible : {e}")

try:
    st.sidebar.caption(f"xgboost: {xgboost.__version__} | sklearn: {sklearn.__version__} | pandas: {pd.__version__}")
except Exception:
    pass