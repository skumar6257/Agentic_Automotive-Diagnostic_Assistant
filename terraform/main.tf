terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "The GCP Project ID"
  type        = string
}

variable "region" {
  description = "The GCP Region"
  type        = string
  default     = "us-central1"
}

# Cloud Run Service for the API Gateway & UI
resource "google_cloud_run_service" "diagnostic_api" {
  name     = "agentic-diagnostic-api"
  location = var.region
  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/diagnostic-api:latest"
        
        ports {
          container_port = 8000
        }

        resources {
          limits = {
            memory = "4Gi"
          }
        }
        
        env {
          name  = "DEPLOYMENT_MODE"
          value = "CLOUD"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Expose the Endpoint Publicly
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.diagnostic_api.name
  location = google_cloud_run_service.diagnostic_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
output "api_url" {
  value = google_cloud_run_service.diagnostic_api.status[0].url
}