# ☁️ Enterprise Cloud Deployment Guide (GCP)

This project features a Dual-Mode architecture (`CLOUD` and `OFFLINE`). This guide outlines the exact steps required to deploy the "Cloud Mode" architecture to Google Cloud using Cloud Build and Terraform.

## 🏗 Architecture Overview
- **Backend API**: FastAPI deployed to Google Cloud Run (Serverless).
- **LLM Engine**: Google Vertex AI (Gemini 1.5 Pro).
- **Graph Database**: Neo4j AuraDB (Cloud Managed).
- **Vector Database**: Qdrant Cloud.
- **CI/CD**: Google Cloud Build (Remote Docker Builds).
- **IaC**: Terraform.

---

## 🛠 Step 1: Google Cloud Console Setup

### 1. Create a Project
- Go to the [Google Cloud Console](https://console.cloud.google.com).
- Create a new project and note your **Project ID**.
- Ensure Billing is enabled.

### 2. Install the Google Cloud CLI
To manage deployments from your terminal, you need the official `gcloud` SDK.
- Download and install the Google Cloud CLI for your OS from the [official site](https://cloud.google.com/sdk/docs/install).
- Restart your terminal (PowerShell or VS Code) to ensure the `gcloud` command is recognized.

### 3. Managed Database Provisioning
This architecture uses managed databases to eliminate infrastructure overhead.
- **Neo4j AuraDB**: Go to [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/), create a free instance, and copy your Connection URI, Username, and Password.
- **Qdrant Cloud**: Go to [Qdrant Cloud](https://cloud.qdrant.io/), create a free cluster, and generate a new API Key. Copy the Cluster URL and API Key.

### 4. Authenticate your Terminal
Open your IDE terminal and securely authenticate using Application Default Credentials (ADC). This prevents you from needing to download risky JSON service account keys!
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 5. Enable Required APIs
Activate the cloud microservices required for this deployment:
```bash
gcloud services enable artifactregistry.googleapis.com run.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com
```

## 🔐 Step 2: Configure Service Account Permissions

When Google Cloud Build runs, it uses the Compute Engine Default Service Account. We must grant it specific roles so it can upload your source code, write to the Artifact Registry, and generate Cloud Logs.

Run these IAM binding commands in your terminal:
```bash
# 1. Allow it to read the temporary source code bucket
gcloud projects add-iam-policy-binding agentic-auto-diag-assistant --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/storage.objectAdmin"

# 2. Allow it to auto-create a new Artifact Registry on push
gcloud projects add-iam-policy-binding agentic-auto-diag-assistant --member="serviceAccount:752600575150-compute@developer.gserviceaccount.com" --role="roles/artifactregistry.createOnPushWriter"

# 3. Allow it to write deployment logs to Cloud Logging
gcloud projects add-iam-policy-binding agentic-auto-diag-assistant --member="serviceAccount:752600575150-compute@developer.gserviceaccount.com" --role="roles/logging.logWriter"

# 4. Allow the Cloud Run service to invoke Vertex AI (Gemini)
gcloud projects add-iam-policy-binding agentic-auto-diag-assistant --member="serviceAccount:752600575150-compute@developer.gserviceaccount.com" --role="roles/aiplatform.user"
```

## ⚙️ Step 3: Application Configuration (.env)
Ensure your .env file is fully configured with your Cloud database URIs and GCP variables. 

**Crucial Step:** When deploying via `gcloud builds submit`, Google Cloud SDK automatically excludes any files listed in your `.gitignore` file (which includes your `.env`!). To bypass this, we have created a `.gcloudignore` file in this project. Because a `.gcloudignore` file is present, `gcloud` will use it instead and securely upload your `.env` file to the remote Docker build environment. 

```env
# Active Deployment Mode
DEPLOYMENT_MODE=CLOUD

GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
AURA_DB_URI=neo4j+s://<your-aura-id>.databases.neo4j.io
AURA_DB_USER=neo4j
AURA_DB_PASSWORD=your-secure-password
QDRANT_CLOUD_URL=https://your-cluster.qdrant.tech:6333
QDRANT_CLOUD_API_KEY=your-qdrant-api-key
```

## 🚀 Step 4: Build & Deploy

### 1. Trigger the Remote CI/CD Pipeline
Instead of building a heavy Docker image locally, we offload it to Google's servers.

```bash
gcloud builds submit --config cloudbuild.yaml .
```

### 2. Deploy Infrastructure via Terraform
Terraform reads terraform/main.tf, configures the server RAM, maps the ports, and deploys the container to the public web.

```bash
cd terraform
terraform init
terraform apply
```

Type yes when prompted. Terraform will output your live $api\_url$.

### 3. Troubleshooting Deployments (Force Update)
Sometimes Terraform may not detect that a new `latest` Docker image was built if the Terraform configuration itself hasn't changed. To forcefully deploy the newly built image to Cloud Run without changing Terraform, run:
```bash
gcloud run services update agentic-diagnostic-api --image gcr.io/YOUR_PROJECT_ID/diagnostic-api:latest --region us-central1 --quiet
```

## 🖥 Step 5: Connect the Streamlit UI

Once your API is live. Replace the CLOUD_API_URL in .env file with your new Cloud Run URL i.e $api\_url$:

```
CLOUD_API_URL=$api_url$/diagnose
```

Run your frontend locally:
```bash
streamlit run frontend/app.py
```