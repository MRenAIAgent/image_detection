.PHONY: deploy-cpu deploy-gpu build-cpu build-gpu test clean

# Default deployment (CPU)
deploy: deploy-cpu

# Build CPU Docker image
build-cpu:
	cd .. && gcloud builds submit --region=${REGION} --config=deploy/cloudbuild.yaml \
		--project=${PROJECT_ID}

# Build GPU Docker image
build-gpu:
	cd .. && gcloud builds submit --region=${REGION} --config=deploy/cloudbuild-gpu.yaml \
		--project=${PROJECT_ID}

# Deploy CPU version to Cloud Run
deploy-cpu-cr:
	gcloud run deploy text-translation-api \
		--project ${PROJECT_ID} \
		--image ${REGION}-docker.pkg.dev/${PROJECT_ID}/text-translation/text-translation-api:latest \
		--memory 16Gi --cpu 4 --min-instances 1 --max-instances 10 \
		--allow-unauthenticated \
		--port 8080 \
		--region ${REGION} \
		--timeout 1800 \
		--concurrency 200 \
		--set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO,DEVICE=cpu,QUANTIZATION_TYPE=int8"
	gcloud --project ${PROJECT_ID} run services update-traffic text-translation-api --to-latest --region ${REGION}

# Deploy GPU version to Cloud Run
deploy-gpu-cr:
	gcloud run deploy text-translation-api-gpu \
		--project ${PROJECT_ID} \
		--image ${REGION}-docker.pkg.dev/${PROJECT_ID}/text-translation/text-translation-api-gpu:latest \
		--memory 16Gi --cpu 4 --min-instances 1 --max-instances 5 \
		--allow-unauthenticated \
		--port 8080 \
		--region ${REGION} \
		--timeout 1800 \
		--concurrency 100 \
		--set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO,DEVICE=cuda,QUANTIZATION_TYPE=fp16" \
		--cpu-boost \
		--execution-environment gen2
	gcloud --project ${PROJECT_ID} run services update-traffic text-translation-api-gpu --to-latest --region ${REGION}

# Full deployment (build + deploy) - CPU
deploy-cpu: build-cpu deploy-cpu-cr
	@echo "CPU deployment completed for project: ${PROJECT_ID}"

# Full deployment (build + deploy) - GPU
deploy-gpu: build-gpu deploy-gpu-cr
	@echo "GPU deployment completed for project: ${PROJECT_ID}"

# Test the deployed API
test:
	@echo "Testing CPU API..."
	@SERVICE_URL=$$(gcloud run services describe text-translation-api --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)") && \
	curl -X POST "$$SERVICE_URL/translate" \
		-H "Content-Type: application/json" \
		-d '{"text": "Hello, world!", "source_language": "english", "target_language": "spanish"}'

# Test GPU API
test-gpu:
	@echo "Testing GPU API..."
	@SERVICE_URL=$$(gcloud run services describe text-translation-api-gpu --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)") && \
	curl -X POST "$$SERVICE_URL/translate" \
		-H "Content-Type: application/json" \
		-d '{"text": "Hello, world!", "source_language": "english", "target_language": "spanish"}'

# Clean up resources
clean:
	@echo "Cleaning up resources..."
	-gcloud run services delete text-translation-api --region=${REGION} --project=${PROJECT_ID} --quiet
	-gcloud run services delete text-translation-api-gpu --region=${REGION} --project=${PROJECT_ID} --quiet
	@echo "Resources cleaned up"