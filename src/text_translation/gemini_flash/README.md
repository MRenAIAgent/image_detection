# Gemini Flash Translation API

A high-performance translation API using Google's Gemini Flash model for fast, accurate text translation.

## 🚀 Features

- **Ultra-fast translations** (200-800ms typical response time)
- **100+ languages** supported
- **Batch processing** for multiple texts
- **Smart caching** to improve performance
- **RESTful API** with comprehensive documentation
- **Zero infrastructure** management (serverless)
- **Auto-scaling** based on demand
- **Built-in monitoring** and statistics

## ⚡ Performance Comparison

| Solution | Latency | Setup | Cost Model |
|----------|---------|-------|------------|
| **Gemini Flash** | **200-800ms** | API only | Pay-per-use |
| NLLB CPU | 2-3s | Full infrastructure | Fixed hosting |
| NLLB GPU | 0.3-0.8s | Full infrastructure | Higher fixed cost |

## 🛠️ Quick Start

### 1. Get Google AI API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Keep it secure - you'll need it for deployment

### 2. Deploy to Cloud Run

```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Deploy the service
cd gemini_flash
./simple_deploy.sh
```

### 3. Set API Key

After deployment, set your Google AI API key:

```bash
gcloud run services update gemini-flash-api \
  --set-env-vars="GOOGLE_API_KEY=your-api-key-here" \
  --region=us-central1 --project=$PROJECT_ID
```

### 4. Test the API

```bash
# Test the deployed service
python test_api.py https://your-service-url.run.app
```

## 📖 API Usage

### Single Translation

```bash
curl -X POST "https://your-service-url/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "source_language": "english",
    "target_language": "spanish"
  }'
```

Response:
```json
{
  "translated_text": "¡Hola, mundo!",
  "source_language": "english",
  "target_language": "spanish",
  "original_text": "Hello, world!",
  "inference_time": 0.456,
  "model": "gemini-1.5-flash",
  "cached": false
}
```

### Batch Translation

```bash
curl -X POST "https://your-service-url/translate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "World", "How are you?"],
    "source_language": "english",
    "target_language": "spanish"
  }'
```

### Supported Languages

```bash
curl "https://your-service-url/languages"
```

## 🌍 Language Support

Supports 100+ languages including:

**Major Languages:**
- English, Spanish, French, German, Italian, Portuguese
- Chinese, Japanese, Korean, Arabic, Hindi, Russian
- Dutch, Swedish, Norwegian, Danish, Finnish

**Language Codes:**
You can use either full names or ISO codes:
- `"english"` or `"en"`
- `"spanish"` or `"es"`
- `"french"` or `"fr"`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/translate` | POST | Single translation |
| `/translate/batch` | POST | Batch translation |
| `/languages` | GET | Supported languages |
| `/model/info` | GET | Model information |
| `/stats` | GET | Usage statistics |
| `/cache/info` | GET | Cache information |
| `/docs` | GET | Interactive API docs |

## ⚙️ Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | **Required** Google AI API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `MAX_TEXT_LENGTH` | `5000` | Maximum text length |
| `MAX_BATCH_SIZE` | `50` | Maximum batch size |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEFAULT_TEMPERATURE` | `0.1` | Model temperature |

## 🔧 Local Development

### Setup

```bash
cd gemini_flash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY=your-api-key
export DEBUG=1
```

### Run Locally

```bash
# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Test locally
python test_api.py http://localhost:8080
```

## 📈 Performance Optimization

### Caching

The API includes intelligent caching:
- **In-memory cache** for repeated translations
- **Automatic cache management** with size limits
- **Cache statistics** available via `/cache/info`

### Batch Processing

For multiple translations:
- Use `/translate/batch` endpoint
- More efficient than individual requests
- Concurrent processing for better performance

### Rate Limiting

Built-in protection:
- Maximum text length: 5,000 characters
- Maximum batch size: 50 texts
- Request timeout: 30 seconds

## 🔍 Monitoring

### Statistics

Get usage statistics:
```bash
curl "https://your-service-url/stats"
```

Returns:
- Total requests and characters processed
- Average inference time
- Error rates
- Cache hit rates

### Health Monitoring

```bash
curl "https://your-service-url/health"
```

Returns service status, uptime, and model information.

## 💰 Cost Estimation

Gemini Flash pricing (approximate):
- **$0.15 per 1M input characters**
- **$0.60 per 1M output characters**

Example costs:
- 1,000 short translations/day: ~$5/month
- 10,000 translations/day: ~$50/month
- 100,000 translations/day: ~$500/month

Plus Cloud Run costs (~$5-20/month for most workloads).

## 🆚 vs NLLB Comparison

| Aspect | Gemini Flash | NLLB |
|--------|-------------|------|
| **Speed** | ⚡ 200-800ms | 🐌 2-3s (CPU) |
| **Setup** | ✅ API only | 🔧 Full infrastructure |
| **Languages** | 🌍 100+ | 🌏 200+ |
| **Context** | 🧠 Excellent | 📖 Good |
| **Cost** | 💵 Pay-per-use | 💰 Fixed hosting |
| **Privacy** | ☁️ Google servers | 🔒 Your infrastructure |
| **Scaling** | 🚀 Automatic | 📈 Manual |

## 🚨 Important Notes

1. **API Key Required**: You must set `GOOGLE_API_KEY` environment variable
2. **Internet Required**: Service needs internet access to reach Google AI API
3. **Rate Limits**: Google AI has usage quotas and rate limits
4. **Data Privacy**: Text is sent to Google's servers for processing

## 🛠️ Troubleshooting

### Common Issues

**1. "Invalid API key" error**
```bash
# Check if API key is set
gcloud run services describe gemini-flash-api --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].env[?name=='GOOGLE_API_KEY'].value)"

# Update API key
gcloud run services update gemini-flash-api \
  --set-env-vars="GOOGLE_API_KEY=your-new-key" \
  --region=us-central1
```

**2. "Service unavailable" error**
- Check Google AI API status
- Verify your API key has sufficient quota
- Check Cloud Run service logs

**3. Slow responses**
- Check your internet connection
- Monitor Google AI API latency
- Consider using caching

### Logs

View service logs:
```bash
gcloud run services logs read gemini-flash-api \
  --region=us-central1 --limit=50
```

## 🔄 Updates and Maintenance

### Update Deployment

```bash
# Redeploy with latest changes
./simple_deploy.sh
```

### Update API Key

```bash
gcloud run services update gemini-flash-api \
  --set-env-vars="GOOGLE_API_KEY=new-key" \
  --region=us-central1
```

### Scale Configuration

```bash
gcloud run services update gemini-flash-api \
  --max-instances=20 \
  --min-instances=1 \
  --region=us-central1
```

## 📚 Additional Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to get started?** Run `./simple_deploy.sh` and have your translation API running in minutes! 🚀