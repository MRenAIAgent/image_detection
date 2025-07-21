.PHONY: deploy_dev

_COMMIT_ID := $(shell git rev-parse HEAD)

cloud_build_docker_dev: PROJECT_ID = "ai-mvp-dv"
cloud_build_docker_dev: REGION = "us-central1"
cloud_build_docker_dev:
	gcloud builds submit --region=${REGION} --config=deploy/cloudbuild.yaml \
		--substitutions _COMMIT_ID="${_COMMIT_ID}" \
		--project=${PROJECT_ID}

deploy_cr_dev: PROJECT_ID = "ai-mvp-dv"
deploy_cr_dev: REGION = "us-central1"
deploy_cr_dev:
	gcloud run deploy copilot-language-server \
		--project ${PROJECT_ID} \
		--image ${REGION}-docker.pkg.dev/${PROJECT_ID}/copilot/copilot-language-server \
		--memory 16Gi --cpu 4 --min-instances 1 \
		--vpc-connector redisvpcconnector \
		--allow-unauthenticated
	gcloud --project ${PROJECT_ID} run services update-traffic copilot-language-server --to-latest

deploy_dev: cloud_build_docker_dev deploy_cr_dev
	@echo "Deploying copilot language server to dev"



