.PHONY: deploy build

# Build Docker image
build:
	cd .. && gcloud builds submit --region=${REGION} --config=deploy/cloudbuild.yaml \
		--project=${PROJECT_ID}

# Deploy to Cloud Run
deploy_cr:
	gcloud run deploy image-recognition-api \
		--project ${PROJECT_ID} \
		--image ${REGION}-docker.pkg.dev/${PROJECT_ID}/image-recognition/image-recognition-api:latest \
		--memory 8Gi --cpu 4 --min-instances 1 \
		--allow-unauthenticated \
		--port 8080 \
		--region ${REGION}
	gcloud --project ${PROJECT_ID} run services update-traffic image-recognition-api --to-latest --region ${REGION}

# Full deployment (build + deploy)
deploy: build deploy_cr
	@echo "Deployment completed for project: ${PROJECT_ID}" 