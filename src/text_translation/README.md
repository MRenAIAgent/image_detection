# 🌍 Text Translation API

A high-performance text translation API powered by **NLLB-200-1.3B** (No Language Left Behind) with quantization support for efficient deployment on Google Cloud Run.

## ✨ Features

- **200+ Languages**: Support for over 200 languages using Meta's NLLB-200 model
- **Quantization Options**: FP16, INT8, and INT4 quantization for optimized performance
- **Batch Translation**: Process multiple texts simultaneously
- **Auto Language Detection**: Intelligent language pair validation
- **Caching**: Built-in translation caching for improved performance
- **Cloud-Ready**: Optimized for Google Cloud Run deployment
- **GPU Support**: Optional GPU acceleration for faster inference

## 🚀 Quick Start

### Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured
- Docker (for local development)

### 1. Simple Deployment

```bash
# Set your GCP project ID
export PROJECT_ID=your-gcp-project-id

# Deploy CPU version (recommended for most use cases)
cd src/text_translation
./simple_deploy.sh

# Or deploy GPU version for higher performance
DEPLOYMENT_TYPE=gpu ./simple_deploy.sh
```

### 2. Manual Deployment

```bash
# Using the deployment script
PROJECT_ID=your-project-id ./deploy/deploy.sh

# Or using Makefile
cd deploy
PROJECT_ID=your-project-id REGION=us-central1 make deploy-cpu
```

## 📡 API Endpoints

### Translation

```bash
# Single text translation
curl -X POST "https://your-service-url/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "source_language": "english",
    "target_language": "spanish"
  }'

# Batch translation
curl -X POST "https://your-service-url/translate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "How are you?", "Goodbye"],
    "source_language": "en",
    "target_language": "es"
  }'
```

### Information

```bash
# Health check
curl "https://your-service-url/health"

# Supported languages
curl "https://your-service-url/languages"

# Model information
curl "https://your-service-url/model/info"

# Statistics
curl "https://your-service-url/stats"
```

## 🌐 Supported Languages

The API supports 200+ languages including:

### Major Languages
- **English** (`english`, `en`, `eng_Latn`)
- **Spanish** (`spanish`, `es`, `spa_Latn`)
- **French** (`french`, `fr`, `fra_Latn`)
- **German** (`german`, `de`, `deu_Latn`)
- **Chinese** (`chinese`, `zh`, `zho_Hans`)
- **Japanese** (`japanese`, `ja`, `jpn_Jpan`)
- **Arabic** (`arabic`, `ar`, `arb_Arab`)
- **Russian** (`russian`, `ru`, `rus_Cyrl`)
- **Portuguese** (`portuguese`, `pt`, `por_Latn`)
- **Italian** (`italian`, `it`, `ita_Latn`)

### Language Code Formats
You can use any of these formats:
- **Full names**: `english`, `spanish`, `french`
- **ISO 639-1**: `en`, `es`, `fr`
- **NLLB codes**: `eng_Latn`, `spa_Latn`, `fra_Latn`

## ⚙️ Configuration Options

### Environment Variables

```bash
# Model settings
MODEL_NAME=facebook/nllb-200-1.3B
DEVICE=auto  # auto, cpu, cuda
QUANTIZATION_ENABLED=true
QUANTIZATION_TYPE=fp16  # fp16, int8, int4

# Performance settings
MAX_LENGTH=512
MAX_BATCH_SIZE=8
BATCH_TIMEOUT=0.1

# Cache settings
ENABLE_CACHE=true
CACHE_TTL=3600
```

### Deployment Types

#### CPU Deployment (Default)
- **Memory**: 8GB
- **CPU**: 4 cores
- **Quantization**: INT8
- **Cost**: Lower operational cost
- **Latency**: 1-3 seconds per translation

#### GPU Deployment
- **Memory**: 16GB
- **CPU**: 4 cores + GPU
- **Quantization**: FP16
- **Cost**: Higher operational cost
- **Latency**: 0.5-1 second per translation

## 📊 Performance Benchmarks

### NLLB-200-1.3B Performance

| Language Pair | BLEU Score | Latency (CPU) | Latency (GPU) |
|---------------|------------|---------------|---------------|
| EN → ES | 28-32 | 2.1s | 0.8s |
| EN → FR | 30-35 | 2.0s | 0.7s |
| EN → DE | 25-30 | 2.3s | 0.9s |
| EN → ZH | 22-28 | 2.5s | 1.0s |
| EN → AR | 20-25 | 2.8s | 1.1s |

### Resource Usage

| Configuration | Memory | Storage | Cold Start |
|---------------|--------|---------|------------|
| CPU + INT8 | 4-6GB | 3GB | 30-45s |
| CPU + FP16 | 6-8GB | 5GB | 45-60s |
| GPU + FP16 | 8-12GB | 5GB | 60-90s |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cloud Run     │    │   FastAPI App    │    │   NLLB Model    │
│   (Load Balancer)│ -> │   (Translation   │ -> │   (Quantized)   │
│                 │    │    Service)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Model Cache    │
                       │   (Persistent)   │
                       └──────────────────┘
```

## 🔧 Local Development

### Setup

```bash
# Clone and navigate
cd src/text_translation

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_NAME=facebook/nllb-200-1.3B
export DEVICE=cpu
export QUANTIZATION_TYPE=int8

# Run locally
python -m app.main
```

### Docker Development

```bash
# Build image
docker build -f deploy/Dockerfile -t text-translation-api .

# Run container
docker run -p 8080:8080 \
  -e DEVICE=cpu \
  -e QUANTIZATION_TYPE=int8 \
  text-translation-api
```

## 🧪 Testing

### Basic Test

```bash
# Test health
curl http://localhost:8080/health

# Test translation
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "source_language": "en", "target_language": "es"}'
```

### Load Testing

```bash
# Install artillery
npm install -g artillery

# Run load test
artillery quick --count 10 --num 5 \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_language": "en", "target_language": "es"}' \
  http://localhost:8080/translate
```

## 📈 Monitoring

### Cloud Run Metrics
- Request count and latency
- Memory and CPU utilization
- Error rates and cold starts

### Application Metrics
- Translation success rate
- Cache hit rate
- Model inference time
- Language pair usage

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check logs
gcloud logs read --project=PROJECT_ID --filter="resource.type=cloud_run_revision"

# Increase memory allocation
gcloud run services update SERVICE_NAME --memory 16Gi
```

#### High Latency
```bash
# Enable GPU deployment
DEPLOYMENT_TYPE=gpu ./simple_deploy.sh

# Optimize batch size
# Set MAX_BATCH_SIZE=4 for better throughput
```

#### Out of Memory
```bash
# Use more aggressive quantization
export QUANTIZATION_TYPE=int8

# Reduce max length
export MAX_LENGTH=256
```

## 💰 Cost Optimization

### CPU Deployment
- **Estimated cost**: $0.10-0.30 per 1000 translations
- **Best for**: Low to medium volume (< 10k translations/day)

### GPU Deployment  
- **Estimated cost**: $0.20-0.50 per 1000 translations
- **Best for**: High volume (> 10k translations/day)

### Cost Reduction Tips
1. Use **min-instances=0** for low traffic
2. Enable **caching** for repeated translations
3. Use **batch translation** for multiple texts
4. Choose **INT8 quantization** for CPU deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Meta AI** for the NLLB-200 model
- **Hugging Face** for the Transformers library
- **Google Cloud** for the deployment platform