# TDS Data Analyst Agent

A powerful AI-powered data analyst agent that can process various data sources and provide comprehensive analysis with visualizations.

## Features

- **Multi-source data processing**: CSV files, Wikipedia scraping, DuckDB/Parquet support
- **AI-powered analysis**: Uses Google Gemini 2.5 Flash for intelligent data analysis
- **Visualization generation**: Creates charts and graphs encoded as base64 images
- **FastAPI endpoint**: RESTful API for easy integration
- **Robust error handling**: Multiple fallback strategies for reliability
- **5-minute timeout protection**: Ensures responses within time limits

## Quick Local Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set your Gemini API key**:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

3. **Start the API server**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```


## Usage

### API Usage

# Local test
curl -X POST "http://localhost:8080/api" -F "file=@question.txt"

# Live deployed API
curl -X POST "https://tds-final-2.onrender.com/api" -F "file=@test_question_1.txt"



### Example Questions

The repository includes three test questions:
- `test_question_1.txt`: Tips dataset analysis
- `test_question_2.txt`: Wikipedia scraping (highest grossing films)
- `test_question_3.txt`: Indian High Court judgments analysis

## API Response Format

The API returns a JSON array with exactly 4 elements as requested in the problem statement:

```json
[
  27,
  "Sun", 
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  0.676
]
```

## Supported Data Sources

- **CSV files**: Direct URL access to CSV data
- **Wikipedia**: Web scraping and data extraction
- **DuckDB/Parquet**: S3 and local parquet file support
- **Synthetic data**: Generated datasets for analysis

## Testing

The implementation has been designed to meet the specific testing requirements:

### Testing Protocol Compliance
- ✅ **5-minute timeout**: API responds within 5 minutes or returns valid JSON error
- ✅ **Always returns JSON**: Never throws HTTP exceptions, always returns valid JSON structure
- ✅ **Retry resilience**: Handles 4 retries with graceful degradation
- ✅ **Multiple simultaneous requests**: Can handle 3 concurrent test cases
- ✅ **Emergency fallbacks**: Returns valid JSON structure even when all processing fails

### Sample Questions Tested
1. **Tips Dataset Analysis**: ✅ Working
2. **Wikipedia Film Data**: ✅ Working  
3. **Indian Court Data**: ✅ Working

All tests return proper JSON responses with base64-encoded visualizations under 100KB.

### Local Testing
```bash
python test_api.py
```

## Deployment

The application is configured for deployment on Render.com with the included `render.yaml` configuration.

## License

MIT License




