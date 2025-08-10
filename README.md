# HuggingFace Model Hash Database Scanner

A comprehensive tool for extracting and verifying file hashes from HuggingFace model repositories. This scanner systematically processes popular models to build a database of file integrity hashes for security verification and change detection.

## Features

- **Multi-method hash extraction**: API metadata, LFS pointer parsing, and direct download
- **Authentication support**: Access gated and private repositories with HuggingFace tokens
- **Failed hash recovery**: Retry failed extractions with authentication
- **Scalable processing**: Concurrent scanning with configurable rate limiting
- **Resume capability**: Restart interrupted scans from where they left off
- **Comprehensive logging**: Detailed error tracking and processing statistics
- **Multiple output formats**: Basic and extended CSV formats with metadata

## Quick Start

### Installation

```bash
pip install huggingface_hub requests tqdm
```

### Basic Usage

```bash
# Scan models with 1000+ downloads
python hf_scanner.py --download-threshold 1000 --output-dir ./output

# Scan with HuggingFace authentication
python hf_scanner.py --token YOUR_HF_API_TOKEN --output-dir ./output

# Update failed hashes with authentication
python hf_scanner.py --token YOUR_HF_API_TOKEN --update-failed-hashes --output-dir ./output
```

## Installation

### Requirements

- Python 3.7+
- HuggingFace Hub access
- Sufficient disk space for temporary downloads

### Dependencies

```txt
huggingface_hub>=0.19.0
requests>=2.28.0
tqdm>=4.64.0
```

### Install Dependencies

```bash
pip install huggingface_hub requests tqdm
```

## Usage

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--download-threshold` | int | 10000 | Minimum download count for models |
| `--output-dir` | str | ./output | Output directory for results |
| `--max-workers` | int | 4 | Maximum concurrent workers |
| `--rate-limit` | float | 1.5 | Delay between requests (seconds) |
| `--max-file-size` | float | 5.0 | Maximum file size to download (GB) |
| `--limit` | int | None | Limit number of models (for testing) |
| `--restart` | flag | False | Restart from most recent files |
| `--restart-from` | str | None | Restart from specific CSV file |
| `--token` | str | None | HuggingFace API token |
| `--update-failed-hashes` | flag | False | Update failed entries with auth |

### Basic Scanning

```bash
# Standard scan
python hf_scanner.py --download-threshold 5000 --output-dir ./results

# High-performance scan
python hf_scanner.py \
    --download-threshold 1000 \
    --max-workers 8 \
    --rate-limit 1.0 \
    --max-file-size 10.0 \
    --output-dir ./large_scan

# Test run with limited models
python hf_scanner.py --download-threshold 1000 --limit 50 --output-dir ./test
```

### Authentication & Private Repositories

```bash
# Scan with authentication
python hf_scanner.py \
    --token hf_your_token_here \
    --download-threshold 1000 \
    --output-dir ./authenticated_scan

# Update failed hashes with authentication
python hf_scanner.py \
    --token hf_your_token_here \
    --update-failed-hashes \
    --output-dir ./authenticated_scan
```

### Resume & Restart

```bash
# Restart from most recent files
python hf_scanner.py --restart --output-dir ./output

# Restart from specific file
python hf_scanner.py --restart-from ./output/hf_model_hashes_extended_20250808_143022.csv

# The restart automatically:
# - Loads previous scan parameters
# - Skips already processed repositories  
# - Continues writing to existing CSV files
```

### Failed Hash Recovery

```bash
# After initial scan, update entries that failed hash extraction
python hf_scanner.py \
    --token hf_your_token_here \
    --update-failed-hashes \
    --output-dir ./output

# This will:
# - Find most recent CSV files automatically
# - Retry FAILED_NO_HASH and FAILED_TOO_LARGE entries
# - Update original CSV files in-place
# - Ignore size limits for authenticated access
```

## Output Format

### File Structure

Each scan produces timestamped files:

```
output/
├── hf_model_hashes_basic_20250808_143022.csv      # Hash, filename, repo_id
├── hf_model_hashes_extended_20250808_143022.csv   # Full metadata
├── hf_scan_errors_20250808_143022.csv             # Error log
└── scan_metadata_20250808_143022.json             # Scan statistics
```

### CSV Format with Embedded Parameters

All CSV files include scan parameters for restart capability:

```csv
# SCAN_PARAMETERS
# download_threshold: 10000
# max_file_size_gb: 5.0
# rate_limit_delay: 1.5
# max_workers: 4
# scan_start: 2025-08-08T14:30:22.123456
# target_extensions: .bin,.caffemodel,.coreml,.dlc,.ggmf,.ggml,.gguf,.ggjt,.h5,.keras,.llamafile,.mar,.mlmodel,.onnx,.pb,.pkl,.pt,.pt2,.pth,.ptl,.safetensors,.tflite
# command_line: python hf_scanner.py --download-threshold 10000 --token [TOKEN_MASKED]
# authenticated: yes
# output_format: extended
# END_PARAMETERS
hash,filename,repo_id,file_size,file_type,last_modified,download_url,hash_method,commit_hash
a1b2c3d4e5f6789...,pytorch_model.bin,microsoft/DialoGPT-medium,345678901,.bin,2023-01-15T10:30:00.000Z,https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin,api_metadata,abc123def456...
```

### Basic CSV Format

Minimal format for hash lookups:

```csv
hash,filename,repo_id
a1b2c3d4e5f6789abcdef123456...,pytorch_model.bin,microsoft/DialoGPT-medium
z9y8x7w6v5u4t3s2r1q0p9o8n7...,model.safetensors,facebook/opt-1.3b
```

### Extended CSV Format

Complete metadata for analysis:

```csv
hash,filename,repo_id,file_size,file_type,last_modified,download_url,hash_method,commit_hash
a1b2c3d4e5f6789...,pytorch_model.bin,microsoft/DialoGPT-medium,345678901,.bin,2023-01-15T10:30:00.000Z,https://...,api_metadata,abc123...
FAILED_TOO_LARGE,model.bin,very-large-model/huge-model,15728640000,.bin,2023-02-01T12:00:00.000Z,https://...,FAILED_TOO_LARGE,
FAILED_NO_HASH,config.json,private/gated-model,1024,.json,2023-01-20T08:30:00.000Z,https://...,FAILED_NO_HASH,
```

### Error CSV Format

Detailed error tracking:

```csv
repo_id,filename,error_type,reason,timestamp
private/model,,gated_repository,Repository requires access request,2025-08-08T14:30:22.123456
huge-model/massive,,file_too_large,File size 12.3GB exceeds limit,2025-08-08T14:31:15.789012
broken/repo,model.bin,hash_extraction_failed,Could not extract hash using any method,2025-08-08T14:32:01.234567
```

### Metadata JSON

Scan statistics and configuration:

```json
{
  "scan_timestamp": "2025-08-08T14:30:22.123456",
  "download_threshold": 10000,
  "max_file_size_gb": 5.0,
  "total_models_scanned": 15234,
  "total_files_processed": 45678,
  "total_errors": 123,
  "target_extensions": [".pt", ".pth", ".onnx", ".safetensors", "..."],
  "rate_limit_delay": 1.5,
  "max_workers": 4,
  "authenticated": true
}
```

## Hash Extraction Methods

The scanner uses a three-tier approach for maximum reliability:

### 1. API Metadata (`api_metadata`)
- **Fastest**: Extract SHA-256 from HuggingFace API responses
- **Most Reliable**: Official metadata from the platform
- **Preferred**: Used when available in file metadata

### 2. LFS Pointer (`lfs_pointer`)
- **Efficient**: Parse Git LFS pointer files for hash information
- **Fallback**: When API metadata unavailable
- **No Download**: Extracts hash without downloading large files

### 3. Direct Download (`downloaded`)
- **Last Resort**: Download file and compute SHA-256 directly
- **Size Limited**: Respects `--max-file-size` parameter
- **Accurate**: Guaranteed correct hash for accessible files

### Special Status Values

- `FAILED_TOO_LARGE`: File exceeds size limit (can be retried with auth)
- `FAILED_NO_HASH`: All extraction methods failed

## Target File Types

The scanner processes model files with these extensions:

| Category | Extensions |
|----------|------------|
| **PyTorch** | `.pt`, `.pth`, `.bin` |
| **ONNX** | `.onnx` |
| **TensorFlow** | `.pb`, `.tflite`, `.h5`, `.keras` |
| **Safetensors** | `.safetensors` |
| **GGML Family** | `.gguf`, `.ggml`, `.ggmf`, `.ggjt` |
| **Apple/Mobile** | `.mlmodel`, `.coreml` |
| **Specialized** | `.llamafile`, `.mar`, `.pt2`, `.ptl` |
| **Other** | `.pkl`, `.caffemodel`, `.dlc` |

## Authentication

### Getting a HuggingFace Token

1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with `read` permissions
3. Use the token with `--token` argument

### Benefits of Authentication

- Access to gated repositories requiring approval
- Access to private repositories you own or have access to
- Higher rate limits for API requests
- Ability to retry large files that may have different access patterns

## Performance & Scaling

### Performance Estimates

Based on real-world testing:

| Metric | Estimate |
|--------|----------|
| Models with 1000+ downloads | ~10,000-50,000 |
| Average files per model | 1-5 |
| Total files to process | ~50,000-250,000 |
| Processing rate | 200-500 files/hour |
| Full scan duration | 4-20 hours |

### Optimization Tips

```bash
# High-throughput configuration
python hf_scanner.py \
    --max-workers 8 \
    --rate-limit 1.0 \
    --max-file-size 2.0 \
    --download-threshold 5000

# Conservative configuration (slower but safer)
python hf_scanner.py \
    --max-workers 2 \
    --rate-limit 3.0 \
    --max-file-size 1.0 \
    --download-threshold 10000
```

### Memory Management

- **Download directory**: Automatically selected (prefers `/tmp`, `/var/tmp`)
- **Cleanup**: Downloaded files are automatically removed after hashing
- **Streaming**: CSV output is written incrementally during processing

## Error Handling & Troubleshooting

### Common Error Types

| Error Type | Description | Solution |
|------------|-------------|----------|
| `repository_not_found` | Repository doesn't exist or is private | Use `--token` for private repos |
| `gated_repository` | Repository requires access approval | Request access on HuggingFace |
| `file_too_large` | File exceeds size limit | Increase `--max-file-size` or use `--update-failed-hashes` |
| `hash_extraction_failed` | All methods failed | Check network/auth, retry with `--update-failed-hashes` |

### Troubleshooting

**Rate Limiting Issues:**
```bash
# Increase delays and reduce workers
python hf_scanner.py --rate-limit 3.0 --max-workers 2
```

**Memory Issues:**
```bash
# Reduce concurrent processing
python hf_scanner.py --max-workers 1 --max-file-size 1.0
```

**Network Timeouts:**
```bash
# The scanner automatically retries failed requests
# Check logs in hf_scanner.log for persistent issues
```

**Authentication Problems:**
```bash
# Verify token has correct permissions
python -c "from huggingface_hub import HfApi; print(HfApi(token='your_token').whoami())"
```

## Example Workflows

### Complete Hash Scan

```bash
# 1. Initial comprehensive scan
python hf_scanner.py \
    --download-threshold 1000 \
    --max-file-size 10.0 \
    --output-dir ./security_scan \
    --token hf_your_token

# 2. Update any failed extractions
python hf_scanner.py \
    --token hf_your_token \
    --update-failed-hashes \
    --output-dir ./security_scan

# 3. Query for specific hash
grep "suspicious_hash_here" ./security_scan/hf_model_hashes_basic_*.csv
```

### Research Dataset

```bash
# Comprehensive research scan
python hf_scanner.py \
    --download-threshold 100 \
    --max-file-size 50.0 \
    --max-workers 6 \
    --output-dir ./research_dataset \
    --token hf_your_token

# Process results for analysis
python -c "
import pandas as pd
df = pd.read_csv('./research_dataset/hf_model_hashes_extended_*.csv', comment='#')
print(f'Total files: {len(df)}')
print(f'Unique repos: {df.repo_id.nunique()}')
print(f'File types: {df.file_type.value_counts()[:10]}')
"
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup

```bash
git clone https://github.com/yourusername/hf-model-hash-scanner
cd hf-model-hash-scanner
pip install -r requirements.txt
```

### Running Tests

```bash
# Test with limited scope
python hf_scanner.py --download-threshold 50000 --limit 10 --output-dir ./test
```

## License

This project is provided as-is for research and security purposes. Please respect HuggingFace's terms of service and rate limits.

## Acknowledgments

- HuggingFace for providing the excellent Hub API
- The open-source ML community for making models accessible
- Contributors to the underlying libraries: `huggingface_hub`, `requests`, `tqdm`