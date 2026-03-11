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
import json
import base64
import time

# --- Config ---
BUCKET = "fruits-bigdata-p11"
REGION = "eu-west-3"
S3_INPUT = "input/images/Training/"
S3_OUTPUT = "output/pca_parquet/parquet_files/"
LAMBDA_URL = "https://5jcr4iliqvarfwuw2h6uca52na0htmit.lambda-url.eu-west-3.on.aws/"

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
    for key in parquet_keys:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        buf = BytesIO(resp["Body"].read())
        df = pq.read_table(buf).to_pandas()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def invoke_lambda(img):
    """Send image to Lambda for inference (MobileNetV2 + PCA + 5-NN)."""
    buf = BytesIO()
    img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    lambda_client = boto3.client(
        "lambda",
        region_name=REGION,
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
    )
    resp = lambda_client.invoke(
        FunctionName="p11-fruit-inference",
        Payload=json.dumps({"image": img_b64}).encode(),
    )
    result = json.loads(resp["Payload"].read())
    if "body" in result:
        return json.loads(result["body"])
    return result


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
    "Pipeline PySpark",
    "Resultats PCA (S3)",
    "Identifier un fruit",
    "Tests de Robustesse",
    "Passage a l'echelle",
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
# PAGE 4: Pipeline PySpark
# ============================================================
elif page == "Pipeline PySpark":
    st.title("Pipeline PySpark - Code et explications")
    st.markdown("Le code PySpark qui a ete execute dans le cloud pour traiter les 67 692 images.")

    st.subheader("Etape 1 : Chargement des images")
    st.code("""
# Chargement des images depuis S3 / stockage cloud
images_df = spark.read.format("image").load(images_path)
print(f"Images chargees: {images_df.count()}")
# -> 67 692 images, 131 categories
    """, language="python")
    st.markdown("Les images sont chargees nativement par Spark au format distribue. "
                "Chaque worker traite un sous-ensemble d'images en parallele.")

    st.markdown("---")
    st.subheader("Etape 2 : Preprocessing")
    st.code("""
# Resize a 224x224 (taille attendue par MobileNetV2)
# Normalisation des pixels [0, 1]
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

preprocess_udf = udf(preprocess_image, ArrayType(FloatType()))
    """, language="python")

    st.markdown("---")
    st.subheader("Etape 3 : Broadcast des poids TensorFlow")
    st.code("""
# Chargement du modele MobileNetV2 (pre-entraine sur ImageNet)
from tensorflow.keras.applications import MobileNetV2
new_model = MobileNetV2(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3))

# BROADCAST : diffusion des poids sur tous les workers du cluster
brodcast_weights = sc.broadcast(new_model.get_weights())
    """, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Sans broadcast :**
- Chaque worker telecharge le modele (~14 MB)
- N workers = N copies en memoire
- Transfert reseau x N
        """)
    with col2:
        st.markdown("""
**Avec broadcast :**
- Le driver envoie les poids une seule fois
- Partages en memoire sur chaque worker
- Optimisation critique pour le calcul distribue
        """)

    st.code("""
def model_fn():
    \"\"\"
    Returns a MobileNetV2 model with top layer removed
    and broadcasted pretrained weights.
    \"\"\"
    model = MobileNetV2(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3))
    model.set_weights(brodcast_weights.value)  # <- poids broadcasted
    return model
    """, language="python")

    st.markdown("---")
    st.subheader("Etape 4 : Extraction de features")
    st.code("""
# Extraction des features avec MobileNetV2
# Output: vecteur de 1280 dimensions par image
features = model.predict(preprocessed_image)
# -> 67 692 images x 1280 features
    """, language="python")

    st.markdown("---")
    st.subheader("Etape 5 : Reduction PCA en PySpark")
    st.code("""
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT

# Conversion des features en format Vector Spark
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
features_df = features_df.withColumn("features_vector",
    list_to_vector_udf("features"))

# PCA distribue : 1280 -> 100 dimensions
n_components = 100
pca = PCA(k=n_components,
          inputCol="features_vector",
          outputCol="pca_features")
pca_model = pca.fit(features_df)

# Transformer les donnees
features_df = pca_model.transform(features_df)

# Variance expliquee
explained_variance = pca_model.explainedVariance
print(f"Variance totale: {sum(explained_variance):.4f}")
# -> ~0.90-0.95 (90-95% de la variance conservee)
    """, language="python")

    st.markdown("---")
    st.subheader("Etape 6 : Sauvegarde sur S3")
    st.code("""
# Sauvegarde en format Parquet distribue sur S3
features_df.select("path", "label", "pca_features") \\
    .write.mode("overwrite") \\
    .parquet("s3://fruits-bigdata-p11/output/pca_parquet/")

# -> 32 fichiers Parquet, 53 MB total
# -> Compression 95% (985 MB -> 53 MB)
    """, language="python")
    st.success("Les sorties sont ecrites directement dans l'espace de stockage cloud S3 "
               "(critere CE3 : ecriture directe sur le cloud).")

    st.markdown("---")
    st.subheader("Preuve d'execution dans le cloud")
    st.markdown("""
**Executions realisees :**

| Date | Environnement | Dataset | Resultat |
|------|--------------|---------|----------|
| 10/12/2024 | Kaggle GPU T4 | 67 692 images | Pipeline complet execute |
| 11/12/2024 | Kaggle GPU T4 | 67 692 images | Classification RF + LR |
| 29/12/2024 | Kaggle GPU T4 | 67 692 images | Classification 3 modeles |

**Logs d'execution (extrait) :**
    """)
    st.code("""
=========================================
PIPELINE COMPLET - 67,692 IMAGES
=========================================
ETAPE 1/4: UPLOAD DATASET
  Upload successful: Training.zip (295MB)
  Dataset disponible: guillaumecassez/fruits-360-full
ETAPE 2/4: PREPARATION DU NOTEBOOK
  Metadonnees creees
ETAPE 3/4: SOUMISSION DU NOTEBOOK
  Kernel version 1 successfully pushed.
  URL: https://www.kaggle.com/code/guillaumecassez/p11-full-dataset-67k
ETAPE 4/4: MONITORING
  Entrainement termine avec succes
  Resultats: 32 fichiers Parquet, 53 MB
    """, language="text")


# ============================================================
# PAGE 5: Resultats PCA depuis S3
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
# PAGE 6: Identifier un fruit
# ============================================================
elif page == "Identifier un fruit":
    st.title("Identification d'un fruit")
    st.markdown("""
    **Pipeline identique au notebook** : MobileNetV2 (1280 features) → PCA (100 dims) → 5-NN cosinus.
    Les references PCA sont chargees depuis S3.
    """)

    # Load a default image from S3 or let user upload
    use_default = st.checkbox("Utiliser une image du dataset S3", value=True)

    user_img = None
    if use_default:
        categories = get_s3_categories()
        if categories:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                default_cat = st.selectbox("Categorie", categories, key="id_cat")
            with col_d2:
                objects = list_s3_objects(f"{S3_INPUT}{default_cat}/", max_keys=20)
                img_keys = [o["Key"] for o in objects if o["Key"].lower().endswith((".jpg", ".png", ".jpeg"))]
                default_idx = st.slider("Image #", 0, max(0, len(img_keys)-1), 0, key="id_idx")
            if img_keys:
                user_img = get_s3_image(img_keys[default_idx]).convert("RGB")
        else:
            st.warning("Aucune image sur S3.")
    else:
        uploaded = st.file_uploader("Prenez en photo un fruit ou uploadez une image",
                                    type=["jpg", "png", "jpeg"], key="identify_upload")
        if uploaded:
            user_img = Image.open(uploaded).convert("RGB")

    if user_img:

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(user_img, caption="Votre image", use_container_width=True)

        with col2:
            st.markdown("**Pipeline AWS Lambda (inference serverless) :**")

            with st.spinner("Envoi a AWS Lambda → MobileNetV2 → PCA → 5-NN cosinus..."):
                start = time.time()
                result = invoke_lambda(user_img)
                elapsed = time.time() - start

            if "error" in result:
                st.error(f"Erreur Lambda: {result['error']}")
            else:
                st.success(f"Inference terminee en {elapsed:.1f}s")

                prediction = result["prediction"]
                confidence = result["confidence"]
                top5 = result["top5"]

                st.markdown("---")
                st.subheader(f"Resultat : {prediction}")

                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Confiance", f"{confidence*100:.1f}%")
                col_m2.metric("Latence inference", f"{elapsed:.1f}s")

                st.markdown("**Top 5 plus proches voisins :**")
                top5_data = []
                for t in top5:
                    top5_data.append({
                        "Rang": t["rank"],
                        "Categorie": t["label"],
                        "Distance cosinus": t["distance"],
                    })
                st.dataframe(pd.DataFrame(top5_data), use_container_width=True, hide_index=True)

                # Bar chart
                fig = px.bar(
                    x=[t["label"] for t in top5],
                    y=[1 - t["distance"] for t in top5],
                    labels={"x": "Categorie", "y": "Similarite (1 - distance)"},
                    title="Similarite cosinus des 5 plus proches voisins"
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

                # Show reference images from best match
                st.markdown(f"**Images de reference ({prediction}) depuis S3 :**")
                ref_objects = list_s3_objects(f"{S3_INPUT}{prediction}/", max_keys=4)
                ref_keys = [o["Key"] for o in ref_objects if o["Key"].lower().endswith((".jpg", ".png", ".jpeg"))]
                if ref_keys:
                    ref_cols = st.columns(min(4, len(ref_keys)))
                    for i, key in enumerate(ref_keys[:4]):
                        with ref_cols[i]:
                            try:
                                st.image(get_s3_image(key), caption=key.split("/")[-1], use_container_width=True)
                            except:
                                pass

                st.markdown("---")
                st.info(
                    "**Architecture** : Streamlit (front-end) → AWS Lambda (inference MobileNetV2 + PCA + 5-NN) "
                    "→ S3 (stockage features PCA et modele). "
                    "Pipeline identique au notebook, resultats coherents."
                )


# ============================================================
# PAGE 7: Tests de Robustesse
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

            # Compute metrics
            st.markdown("---")
            st.subheader("Metriques de distorsion")

            orig_arr = np.array(original_rgb).astype(np.float64) / 255.0
            rows_metrics = []
            for name, img in transforms.items():
                if name == "Original":
                    continue
                t_arr = np.array(img.resize(original_rgb.size)).astype(np.float64) / 255.0
                mse = np.mean((orig_arr - t_arr) ** 2)
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                # Correlation on flattened grayscale
                gray_o = np.mean(orig_arr, axis=2).flatten()
                gray_t = np.mean(t_arr, axis=2).flatten()
                corr = np.corrcoef(gray_o, gray_t)[0, 1]
                # Color distance (mean absolute diff per channel)
                color_diff = np.mean(np.abs(orig_arr - t_arr)) * 255
                rows_metrics.append({
                    "Transformation": name,
                    "MSE": round(mse, 5),
                    "PSNR (dB)": round(psnr, 1),
                    "Correlation": round(corr, 4),
                    "Ecart couleur moyen": round(color_diff, 1),
                    "Impact": "Fort" if corr < 0.85 else ("Moyen" if corr < 0.95 else "Faible"),
                })

            df_metrics = pd.DataFrame(rows_metrics)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            st.markdown("""
**Legende :**
- **MSE** : erreur quadratique moyenne (0 = identique)
- **PSNR** : ratio signal/bruit en dB (plus haut = plus proche de l'original)
- **Correlation** : similarite structurelle (1.0 = identique)
- **Ecart couleur** : difference moyenne par pixel (0-255)
- **Impact** : Fort (corr < 0.85), Moyen (< 0.95), Faible (>= 0.95)
            """)

            # Bar chart
            fig = px.bar(
                df_metrics, x="Transformation", y="Correlation",
                title="Correlation avec l'image originale par transformation",
                color="Impact",
                color_discrete_map={"Faible": "#2ecc71", "Moyen": "#f39c12", "Fort": "#e74c3c"}
            )
            fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

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
# PAGE 8: Passage a l'echelle
# ============================================================
elif page == "Passage a l'echelle":
    st.title("Passage a l'echelle et retour critique")

    st.subheader("Traitements critiques identifies")
    st.markdown("""
| Traitement | Temps actuel (67k) | Projection (1M images) | Bottleneck | Solution |
|-----------|-------------------|----------------------|-----------|----------|
| **Chargement images** | ~5 min | ~1h15 | I/O disque, reseau | Partitionnement S3, lecture parallele |
| **Preprocessing** | ~10 min | ~2h30 | CPU-bound | Distribution sur workers EMR |
| **Feature extraction** | ~1h (GPU) | ~15h (1 GPU) | GPU-bound | Multi-GPU, instances p3/g4dn |
| **Broadcast poids** | ~5s | ~5s | Stable | Une seule copie, invariant au volume |
| **PCA fit** | ~3 min | ~45 min | Memoire driver | PCA distribue PySpark (deja en place) |
| **PCA transform** | ~2 min | ~30 min | CPU workers | Ajout de workers spot |
| **Ecriture Parquet** | ~1 min | ~15 min | I/O S3 | Partitionnement, ecriture parallele |
    """)

    st.markdown("---")
    st.subheader("Strategie de scalabilite")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Scalabilite horizontale (EMR)**

- Ajouter des workers au cluster
- Instances spot (-60 a -70% de cout)
- Auto-scaling selon la charge
- PySpark distribue automatiquement

**Exemple de configuration :**
- 67k images : 1 master + 2 workers
- 500k images : 1 master + 8 workers
- 1M+ images : 1 master + 20 workers spot
        """)
    with col2:
        st.markdown("""
**Optimisations identifiees**

- **Partitionnement S3** : organiser par categorie
- **Format Parquet** : lecture partielle (column pruning)
- **Cache Spark** : persister les DataFrames intermediaires
- **Broadcast** : deja en place, invariant au volume
- **GPU cloud** : instances g4dn pour feature extraction
- **Compression Snappy** : reduit I/O de 95%
        """)

    st.markdown("---")
    st.subheader("Retour critique sur la solution")

    st.markdown("""
**Points forts :**
- Pipeline PySpark reproductible et distribue
- Format Parquet optimise pour le Big Data
- Broadcast des poids = optimisation memoire significative
- PCA distribue = scalable nativement
- Architecture cloud RGPD conforme

**Limites identifiees :**
- Feature extraction encore CPU/GPU-bound (pas distribue dans Spark nativement)
- MobileNetV2 pre-entraine sur ImageNet (pas sur des fruits specifiquement)
- Dataset Fruits-360 = conditions ideales (fond blanc, eclairage uniforme)
- PCA lineaire : une reduction non-lineaire (UMAP, autoencoder) pourrait mieux capturer les variations

**Recommandations avant generalisation :**
- Fine-tuning de MobileNetV2 sur le dataset Fruits-360
- Augmentation du dataset avec images reelles (conditions terrain)
- Tests de performance avec volumes croissants (100k, 500k, 1M)
- Mise en place du monitoring des couts cloud
- Pipeline CI/CD pour re-entrainement automatique
    """)

    st.markdown("---")
    st.subheader("Estimation des couts")

    st.markdown("""
| Volume | Config EMR | Temps estime | Cout estime |
|--------|-----------|-------------|-------------|
| 67k images (actuel) | 1 master + 1 worker spot | ~2h | ~0.80 EUR |
| 500k images | 1 master + 4 workers spot | ~4h | ~4.00 EUR |
| 1M images | 1 master + 8 workers spot | ~6h | ~12.00 EUR |
| 5M images | 1 master + 20 workers spot | ~8h | ~40.00 EUR |

*Cout base sur m5.large on-demand (master) + m5.xlarge spot (workers), region eu-west-3*
    """)


# ============================================================
# PAGE 9: Architecture
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
