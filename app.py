"""
Application Streamlit - Demo AWS S3 Integration
Projet P11 Big Data Cloud - Fruits Recognition
Deployee sur Streamlit Community Cloud
"""

import streamlit as st
import boto3
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
import time

# --- Config ---
BUCKET = "fruits-bigdata-p11"
REGION = "eu-west-3"
S3_INPUT = "input/images/Training/"
S3_OUTPUT = "output/pca_parquet/parquet_files/"

st.set_page_config(
    page_title="P11 - Demo AWS Big Data",
    page_icon="☁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- S3 Client (credentials from Streamlit secrets) ---
@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
    )


def get_sts_client():
    return boto3.client(
        "sts",
        region_name=REGION,
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
    )


# --- S3 Helper Functions ---
@st.cache_data(ttl=60)
def list_s3_objects(prefix, max_keys=1000):
    s3 = get_s3_client()
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=max_keys)
        return resp.get("Contents", [])
    except Exception as e:
        st.error(f"Erreur S3: {e}")
        return []


@st.cache_data(ttl=60)
def get_s3_categories():
    s3 = get_s3_client()
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=S3_INPUT, Delimiter="/")
        prefixes = resp.get("CommonPrefixes", [])
        return [p["Prefix"].rstrip("/").split("/")[-1] for p in prefixes]
    except Exception as e:
        st.error(f"Erreur S3: {e}")
        return []


def get_s3_image(key):
    s3 = get_s3_client()
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return Image.open(BytesIO(resp["Body"].read()))


@st.cache_data(ttl=300)
def load_pca_from_s3():
    s3 = get_s3_client()
    objects = list_s3_objects(S3_OUTPUT)
    parquet_keys = [o["Key"] for o in objects if o["Key"].endswith(".parquet")]
    if not parquet_keys:
        return None
    dfs = []
    for key in parquet_keys[:5]:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        buf = BytesIO(resp["Body"].read())
        df = pq.read_table(buf).to_pandas()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_bucket_info():
    s3 = get_s3_client()
    try:
        total_size = 0
        total_count = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, PaginationConfig={"MaxItems": 5000}):
            for obj in page.get("Contents", []):
                total_size += obj["Size"]
                total_count += 1
        return total_count, total_size
    except Exception as e:
        return 0, 0


# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", [
    "Connexion AWS",
    "S3 - Stockage Cloud",
    "Images sur S3",
    "Resultats PCA (S3)",
    "Tests de Robustesse",
    "Architecture",
    "Perspectives"
])

st.sidebar.markdown("---")
st.sidebar.caption(f"Bucket: `{BUCKET}`")
st.sidebar.caption(f"Region: `{REGION}` (Paris - RGPD)")


# ============================================================
# PAGE 1: Connexion AWS
# ============================================================
if page == "Connexion AWS":
    st.title("Connexion AWS - Verification Live")
    st.markdown("Cette page demontre la connexion en temps reel avec les services AWS depuis Streamlit Cloud.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test de connexion S3")
        if st.button("Tester la connexion", type="primary"):
            with st.spinner("Connexion a AWS S3..."):
                start = time.time()
                try:
                    s3 = get_s3_client()
                    s3.head_bucket(Bucket=BUCKET)
                    elapsed = time.time() - start
                    st.success(f"Connexion reussie en {elapsed:.2f}s")

                    location = s3.get_bucket_location(Bucket=BUCKET)
                    loc = location["LocationConstraint"] or "us-east-1"

                    st.markdown(f"""
| Parametre | Valeur |
|-----------|--------|
| **Bucket** | `{BUCKET}` |
| **Region** | `{loc}` |
| **RGPD** | {"Conforme (EU)" if "eu-" in loc else "Non-EU"} |
| **Latence** | {elapsed*1000:.0f} ms |
                    """)
                except Exception as e:
                    st.error(f"Echec: {e}")

    with col2:
        st.subheader("Informations du compte")
        try:
            sts = get_sts_client()
            identity = sts.get_caller_identity()
            st.markdown(f"""
| Parametre | Valeur |
|-----------|--------|
| **Account** | `{identity['Account']}` |
| **User** | `{identity['Arn'].split('/')[-1]}` |
| **Region** | `{REGION}` (Paris) |
            """)
        except Exception as e:
            st.warning(f"Impossible de recuperer l'identite: {e}")

    st.markdown("---")
    st.subheader("Architecture deployee")
    st.code("""
Streamlit Cloud (cette application)
     |
     | boto3 (credentials securises)
     v
AWS S3 (eu-west-3, Paris)
     |
     +-- input/images/Training/    <- Dataset Fruits-360
     |       (131 categories, 67k images)
     |
     +-- output/pca_parquet/       <- Resultats PCA
            (32 fichiers Parquet, 53 MB)
            (67k lignes x 100 dimensions)
    """, language="text")


# ============================================================
# PAGE 2: S3 Stockage
# ============================================================
elif page == "S3 - Stockage Cloud":
    st.title("AWS S3 - Exploration du Stockage Cloud")

    with st.spinner("Chargement des informations S3..."):
        total_count, total_size = get_bucket_info()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fichiers sur S3", f"{total_count:,}")
    col2.metric("Taille totale", f"{total_size / (1024**2):.1f} MB")
    col3.metric("Bucket", BUCKET)
    col4.metric("Region", "eu-west-3 (Paris)")

    st.markdown("---")
    st.subheader("Arborescence S3")
    tab1, tab2 = st.tabs(["Images (input)", "Resultats PCA (output)"])

    with tab1:
        categories = get_s3_categories()
        if categories:
            st.write(f"**{len(categories)} categories** de fruits sur S3 :")
            for cat in categories:
                objects = list_s3_objects(f"{S3_INPUT}{cat}/", max_keys=5)
                count = len(objects)
                st.markdown(f"- `{cat}/` ({count}+ images)")
        else:
            st.info("Aucune image uploadee sur S3 pour le moment.")

    with tab2:
        parquet_files = list_s3_objects(S3_OUTPUT)
        parquet_only = [o for o in parquet_files if o["Key"].endswith(".parquet")]
        if parquet_only:
            st.write(f"**{len(parquet_only)} fichiers Parquet** sur S3 :")
            df_files = pd.DataFrame([
                {
                    "Fichier": o["Key"].split("/")[-1],
                    "Taille (KB)": round(o["Size"] / 1024, 1),
                    "Modifie": o["LastModified"].strftime("%Y-%m-%d %H:%M")
                }
                for o in parquet_only
            ])
            st.dataframe(df_files, use_container_width=True, hide_index=True)
            total_parquet = sum(o["Size"] for o in parquet_only)
            st.metric("Taille totale Parquet", f"{total_parquet / (1024**2):.1f} MB")
        else:
            st.info("Aucun fichier Parquet trouve.")


# ============================================================
# PAGE 3: Images sur S3
# ============================================================
elif page == "Images sur S3":
    st.title("Images de Fruits sur AWS S3")
    st.markdown("Chargement direct des images depuis le bucket S3.")

    categories = get_s3_categories()
    if not categories:
        st.warning("Aucune categorie d'images sur S3.")
        st.stop()

    selected_cat = st.selectbox("Categorie de fruit", categories)

    if selected_cat:
        objects = list_s3_objects(f"{S3_INPUT}{selected_cat}/")
        image_keys = [o["Key"] for o in objects if o["Key"].lower().endswith((".jpg", ".png", ".jpeg"))]
        st.write(f"**{len(image_keys)} images** dans `{selected_cat}/`")

        n_show = min(12, len(image_keys))
        cols = st.columns(4)
        for i in range(n_show):
            with cols[i % 4]:
                try:
                    img = get_s3_image(image_keys[i])
                    st.image(img, caption=image_keys[i].split("/")[-1], use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur: {e}")

    st.markdown("---")
    st.subheader("Upload d'image vers S3")
    uploaded = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption=uploaded.name, width=200)
        if st.button("Envoyer vers S3", type="primary"):
            s3 = get_s3_client()
            key = f"input/images/uploads/{uploaded.name}"
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
            st.success(f"Image uploadee vers `s3://{BUCKET}/{key}`")


# ============================================================
# PAGE 4: Resultats PCA depuis S3
# ============================================================
elif page == "Resultats PCA (S3)":
    st.title("Resultats PCA - Lecture depuis AWS S3")
    st.markdown("Les donnees sont lues **en temps reel** depuis le bucket S3.")

    with st.spinner("Chargement des fichiers Parquet depuis S3..."):
        start = time.time()
        df = load_pca_from_s3()
        elapsed = time.time() - start

    if df is None:
        st.error("Aucun fichier Parquet trouve sur S3.")
        st.stop()

    st.success(f"Charge {len(df):,} lignes en {elapsed:.2f}s depuis S3")

    col1, col2, col3 = st.columns(3)
    col1.metric("Echantillons charges", f"{len(df):,}")
    col2.metric("Colonnes", len(df.columns))
    col3.metric("Source", f"s3://{BUCKET}/...")

    st.markdown("---")
    st.subheader("Apercu des donnees")
    st.dataframe(df.head(20), use_container_width=True)

    if "label" in df.columns:
        st.markdown("---")
        st.subheader("Distribution des categories")
        label_counts = df["label"].value_counts().head(20)
        fig = px.bar(
            x=label_counts.index, y=label_counts.values,
            labels={"x": "Categorie", "y": "Nombre d'images"},
            title="Top 20 categories (echantillon S3)"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    if "pca_features" in df.columns:
        st.markdown("---")
        st.subheader("Visualisation PCA 2D")
        try:
            pca_data = df["pca_features"].apply(
                lambda x: x["values"] if isinstance(x, dict) and "values" in x
                else (list(x) if hasattr(x, "__iter__") else x)
            )
            pca_matrix = np.array(pca_data.tolist())

            if pca_matrix.ndim == 2 and pca_matrix.shape[1] >= 2:
                from sklearn.decomposition import PCA as SKPCA
                pca_2d = SKPCA(n_components=2)
                coords = pca_2d.fit_transform(pca_matrix)

                df_viz = pd.DataFrame({
                    "PC1": coords[:, 0], "PC2": coords[:, 1],
                    "Label": df["label"] if "label" in df.columns else "unknown"
                })
                top_labels = df_viz["Label"].value_counts().head(15).index
                df_viz_filtered = df_viz[df_viz["Label"].isin(top_labels)]

                fig = px.scatter(
                    df_viz_filtered, x="PC1", y="PC2", color="Label",
                    title="PCA 2D - Clusters de fruits (donnees S3)", opacity=0.6
                )
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Variance expliquee (2 composantes)",
                          f"{pca_2d.explained_variance_ratio_.sum()*100:.1f}%")
        except Exception as e:
            st.warning(f"Visualisation PCA non disponible: {e}")


# ============================================================
# PAGE 5: Tests de Robustesse
# ============================================================
elif page == "Tests de Robustesse":
    st.title("Tests de Robustesse du Pipeline")
    st.markdown("Evaluation de la robustesse face a des images modifiees : "
                "crop, deformations, flou, rotation...")

    tab1, tab2 = st.tabs(["Transformations sur images S3", "Test avec image externe"])

    with tab1:
        categories = get_s3_categories()
        if not categories:
            st.warning("Aucune image sur S3.")
            st.stop()

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            selected_cat = st.selectbox("Categorie", categories, key="robust_cat")
        with col_sel2:
            objects = list_s3_objects(f"{S3_INPUT}{selected_cat}/")
            image_keys = [o["Key"] for o in objects if o["Key"].lower().endswith((".jpg", ".png", ".jpeg"))]
            selected_idx = st.slider("Image #", 0, max(0, len(image_keys)-1), 0)

        if image_keys:
            original = get_s3_image(image_keys[selected_idx])
            original_rgb = original.convert("RGB")
            w, h = original_rgb.size

            transforms = {}
            transforms["Original"] = original_rgb
            transforms["Crop 50% (haut-gauche)"] = original_rgb.crop((0, 0, w//2, h//2)).resize((w, h))
            margin_w, margin_h = w//4, h//4
            transforms["Crop centre (zoom)"] = original_rgb.crop(
                (margin_w, margin_h, w - margin_w, h - margin_h)).resize((w, h))
            transforms["Rotation 45deg"] = original_rgb.rotate(45, expand=False, fillcolor=(0, 0, 0))
            transforms["Miroir horizontal"] = original_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            transforms["Flou gaussien"] = original_rgb.filter(ImageFilter.GaussianBlur(radius=5))
            transforms["Sombre (-50%)"] = ImageEnhance.Brightness(original_rgb).enhance(0.5)
            transforms["Desature"] = ImageEnhance.Color(original_rgb).enhance(0.2)
            noisy = np.array(original_rgb).copy()
            noise_mask = np.random.random(noisy.shape[:2])
            noisy[noise_mask < 0.05] = 0
            noisy[noise_mask > 0.95] = 255
            transforms["Bruit (salt & pepper)"] = Image.fromarray(noisy)

            st.markdown("---")
            st.subheader("Comparaison des transformations")
            items = list(transforms.items())
            for row_start in range(0, len(items), 3):
                cols = st.columns(3)
                for i, (name, img) in enumerate(items[row_start:row_start+3]):
                    with cols[i]:
                        st.image(img, caption=name, use_container_width=True)

            st.markdown("---")
            st.subheader("Analyse de l'impact")
            st.markdown("""
| Transformation | Impact attendu | Strategie |
|---------------|----------------|-----------|
| **Crop partiel** | Fort - perte de forme globale | Data augmentation avec crops |
| **Rotation** | Moyen - MobileNetV2 partiellement invariant | Augmentation rotations |
| **Miroir** | Faible - symetrie preservee | Flip horizontal standard |
| **Flou** | Moyen - perte de texture | Simule photos floues reelles |
| **Sombre** | Moyen - perte de contraste | Normalisation + augmentation |
| **Desature** | Fort - couleur discriminante | Travailler aussi sur la forme |
| **Bruit** | Faible a moyen | Denoising en preprocessing |
            """)

    with tab2:
        st.subheader("Test avec une image externe")
        st.markdown("Uploadez une image de fruit (photo reelle, capture Auchan Drive, etc.)")

        uploaded = st.file_uploader("Image a tester", type=["jpg", "png", "jpeg"], key="robust_upload")
        if uploaded:
            test_img = Image.open(uploaded).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.image(test_img, caption="Image uploadee", use_container_width=True)
                st.markdown(f"**Taille** : {test_img.size[0]}x{test_img.size[1]} px")

            with col2:
                st.markdown("**Comparaison avec le dataset Fruits-360 :**")
                st.markdown(f"- Dataset : images 100x100 px, fond blanc uniforme")
                st.markdown(f"- Votre image : {test_img.size[0]}x{test_img.size[1]} px")

                corners = [
                    test_img.getpixel((0, 0)),
                    test_img.getpixel((test_img.size[0]-1, 0)),
                    test_img.getpixel((0, test_img.size[1]-1)),
                    test_img.getpixel((test_img.size[0]-1, test_img.size[1]-1))
                ]
                avg_brightness = np.mean(corners)
                if avg_brightness > 200:
                    st.success("Fond clair detecte - similaire au dataset")
                else:
                    st.warning("Fond non-blanc - performances potentiellement reduites")

            st.markdown("---")
            st.subheader("Pipeline de preprocessing")
            prep_cols = st.columns(3)
            with prep_cols[0]:
                st.image(test_img, caption="1. Original", use_container_width=True)
            resized = test_img.resize((224, 224))
            with prep_cols[1]:
                st.image(resized, caption="2. Resize 224x224", use_container_width=True)
            arr = np.array(resized).astype(np.float32) / 255.0
            with prep_cols[2]:
                st.image(resized, caption="3. Normalise [0,1]", use_container_width=True)
                st.caption(f"Mean: {arr.mean():.3f}, Std: {arr.std():.3f}")

            st.info("En production, ce vecteur serait extrait par MobileNetV2 (1280 dims) "
                    "puis projete par PCA (100 dims) pour classification.")


# ============================================================
# PAGE 6: Architecture
# ============================================================
elif page == "Architecture":
    st.title("Architecture Big Data Cloud")

    st.subheader("Pipeline de traitement")
    st.code("""
1. INGESTION                    2. STOCKAGE S3              3. TRAITEMENT
+-----------------+            +------------------+         +------------------+
| Dataset local   |  -------> | s3://fruits-      |  ----> | PySpark (EMR/    |
| 67k images      |  AWS CLI  | bigdata-p11/      |        | Colab/Kaggle)    |
| 985 MB          |           | input/images/     |        |                  |
+-----------------+            +------------------+         +------------------+
                                                                   |
                                                                   v
5. VISUALISATION               4. RESULTATS S3
+-----------------+            +------------------+
| Streamlit Cloud |  <------- | s3://fruits-      |
| (cette app)     |  boto3    | bigdata-p11/      |
| Plotly/pandas   |           | output/pca_parquet |
+-----------------+            +------------------+
    """, language="text")

    st.markdown("---")
    st.subheader("Services AWS utilises")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
#### S3 (Storage)
- Stockage distribue
- Region : eu-west-3 (Paris)
- RGPD conforme
- Format Parquet (Snappy)
        """)
    with col2:
        st.markdown("""
#### EMR (Compute)
- Cluster Spark manage
- PySpark 3.5
- Instances m5.large (spot)
- MobileNetV2 + PCA
        """)
    with col3:
        st.markdown("""
#### IAM (Securite)
- User dedie + groupe
- Policies ciblees
- Chiffrement SSE-S3 + TLS
- CloudTrail
        """)

    st.markdown("---")
    st.subheader("Metriques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images traitees", "67,692")
    col2.metric("Categories", "131")
    col3.metric("Reduction PCA", "92.2%", "1280 -> 100 dims")
    col4.metric("Compression", "95%", "985 MB -> 53 MB")

    st.markdown("---")
    st.subheader("Conformite RGPD")
    st.markdown("""
| Critere | Implementation |
|---------|---------------|
| **Localisation** | eu-west-3 (Paris, France) |
| **Transfert** | Intra-EU uniquement |
| **Chiffrement** | SSE-S3 (repos), TLS (transit) |
| **Acces** | IAM User + Groupe |
| **Tracabilite** | CloudTrail |
    """)


# ============================================================
# PAGE 7: Perspectives
# ============================================================
elif page == "Perspectives":
    st.title("Perspectives d'Evolution")

    st.subheader("1. Robustesse de la detection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Conditions reelles vs entrainement**

- **Fruit partiel** : crops aleatoires a l'entrainement
- **Maturite** : detecter vert/jaune/marron
- **Images reelles** : Auchan Drive, marche, verger
- **Fond variable** : segmentation prealable
        """)
    with col2:
        st.markdown("""
**Detection avancee**

- **Imagerie UV** : maladies invisibles a l'oeil nu
- **Insectes** : detection et classification (YOLO)
- **Multi-fruits** : Faster R-CNN
- **Score de confiance** dans la prediction
        """)

    st.markdown("---")
    st.subheader("2. Architecture production")
    st.code("""
[App Mobile User A] ---> [API Gateway] ---> [S3 /user_A/images/]
[App Mobile User B] ---> [API Gateway] ---> [S3 /user_B/images/]
                                                     |
                              Pipeline isole par user (pas de contamination)
                                                     |
                                                     v
                              [EMR / SageMaker] --> Entrainement centralise
                                                     |
                                                     v
                              [MLflow] --> Versioning modeles --> Deploy TFLite/ONNX
                                                                      |
                                                                      v
                                                              Inference locale
                                                              (faible latence, offline)
    """, language="text")

    st.markdown("---")
    st.subheader("3. Gestion des modeles")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Versioning et CI/CD**

- MLflow / DVC pour tracker les versions
- Rollback en cas de regression
- Validation automatique des nouvelles images
- Detection de drift, re-entrainement auto
        """)
    with col2:
        st.markdown("""
**Deploiement edge**

- Entrainement centralise (GPU serveur)
- Export optimise (TFLite, ONNX)
- Inference locale sur mobile
- Monitoring latence par device
        """)
