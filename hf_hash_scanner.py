#!/usr/bin/env python3
"""
HuggingFace Model Hash Database Scanner

This script scans HuggingFace models above a download threshold and extracts
file hashes for integrity verification. It supports both metadata extraction
and file downloading for hash computation.

Usage Examples:
    python hf_scanner.py --download-threshold 1000 --output-dir ./output
    python hf_scanner.py --update --previous-scan ./output/hf_model_hashes_basic.csv
    python hf_scanner.py --token HUGGINGFACE_API_TOKEN --update-failed-hashes --output-dir ./output --max-file-size 5.0
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

import requests
from huggingface_hub import HfApi, hf_hub_download, model_info, login
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hf_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Target file extensions for model files
TARGET_EXTENSIONS = {
    '.pt', '.pth', '.onnx', '.tflite', '.pb', '.safetensors', '.bin',
    '.mlmodel', '.gguf', '.h5', '.mar', '.pt2', '.ptl', '.keras',
    '.ggml', '.ggmf', '.ggjt', '.coreml', '.llamafile', '.pkl',
    '.caffemodel', '.dlc'
}

@dataclass
class FileInfo:
    """Information about a model file"""
    hash: str
    filename: str
    repo_id: str
    file_size: int
    file_type: str
    last_modified: str
    download_url: str
    hash_method: str  # 'api_metadata', 'lfs_pointer', 'downloaded', 'FAILED_TOO_LARGE', 'FAILED_NO_HASH'
    commit_hash: str
    
    def to_basic_dict(self) -> Dict:
        """Return basic format dictionary"""
        return {
            'hash': self.hash,
            'filename': self.filename,
            'repo_id': self.repo_id
        }
    
    def to_extended_dict(self) -> Dict:
        """Return extended format dictionary"""
        return asdict(self)

@dataclass 
class ErrorInfo:
    """Information about scanning errors"""
    repo_id: str
    filename: str
    error_type: str
    reason: str
    timestamp: str

class HuggingFaceScanner:
    """Main scanner class for HuggingFace models"""
    
    def __init__(self, download_threshold: int = 1000, max_workers: int = 4, 
                 rate_limit_delay: float = 1.5, max_file_size_gb: float = 5.0,
                 token: Optional[str] = None):
        self.api = HfApi(token=token)
        self.download_threshold = download_threshold
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_file_size_bytes = int(max_file_size_gb * 1024 * 1024 * 1024)
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HF-Hash-Scanner/1.0 (Research Tool)'
        })
        
        # Add auth header if token provided
        if token:
            self.session.headers.update({
                'Authorization': f'Bearer {token}'
            })
            # Login to huggingface_hub
            login(token=token)
            logger.info("Authenticated with HuggingFace using provided token")
        
        # Setup download directory on non-SSD storage if possible
        self.download_dir = self._setup_download_directory()
        
        # Tracking
        self.processed_models = 0
        self.processed_files = 0
        self.errors: List[ErrorInfo] = []
        
        # CSV writers for streaming output
        self.basic_writer = None
        self.extended_writer = None
        self.error_writer = None
        self.basic_file = None
        self.extended_file = None
        self.error_file = None
        
    def _setup_download_directory(self) -> str:
        """Setup download directory preferring non-SSD storage"""
        possible_dirs = [
            '/tmp',  # Linux temp (often tmpfs, but configurable)
            '/var/tmp',  # Linux persistent temp
            os.path.expanduser('~/Downloads'),  # User downloads
            tempfile.gettempdir()  # System default
        ]
        
        # Try to find a directory that's likely on HDD
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
                # Create subdirectory for our downloads
                download_path = os.path.join(dir_path, 'hf_scanner_downloads')
                os.makedirs(download_path, exist_ok=True)
                logger.info(f"Using download directory: {download_path}")
                return download_path
        
        # Fallback to current directory
        download_path = './hf_scanner_downloads'
        os.makedirs(download_path, exist_ok=True)
        logger.warning(f"Using fallback download directory: {download_path}")
        return download_path
    
    def _get_file_size_from_api(self, repo_id: str, filename: str) -> Optional[int]:
        """Get file size from HuggingFace API"""
        try:
            info = model_info(repo_id, token=self.token)
            for file_info in info.siblings:
                if file_info.rfilename == filename:
                    # Try LFS info first
                    if hasattr(file_info, 'lfs') and file_info.lfs and 'size' in file_info.lfs:
                        return file_info.lfs['size']
                    # Fallback to size attribute if available
                    if hasattr(file_info, 'size') and file_info.size:
                        return file_info.size
        except Exception as e:
            logger.debug(f"Could not get file size for {repo_id}/{filename}: {e}")
        return None
    
    def log_error(self, repo_id: str, filename: str, error_type: str, reason: str):
        """Log an error for later output"""
        error = ErrorInfo(
            repo_id=repo_id,
            filename=filename,
            error_type=error_type,
            reason=reason,
            timestamp=datetime.now().isoformat()
        )
        self.errors.append(error)
        
        # Write to error CSV immediately
        if self.error_writer:
            self.error_writer.writerow(asdict(error))
            self.error_file.flush()
        
        logger.error(f"Error processing {repo_id}/{filename}: {error_type} - {reason}")
    
    def get_models_above_threshold(self, limit: Optional[int] = None) -> List[Dict]:
        """Get models with downloads above threshold"""
        logger.info(f"Fetching models with downloads >= {self.download_threshold}")
        
        all_models = []
        try:
            # Get models sorted by downloads, descending
            models = self.api.list_models(
                sort="downloads",
                direction=-1,
                limit=limit,
                full=True
            )
            
            for model in tqdm(models, desc="Filtering models by download threshold"):
                if hasattr(model, 'downloads') and model.downloads and model.downloads >= self.download_threshold:
                    all_models.append({
                        'repo_id': model.modelId,
                        'downloads': model.downloads,
                        'last_modified': model.lastModified.isoformat() if model.lastModified else None,
                        'sha': getattr(model, 'sha', None)
                    })
                    
                    # Add rate limiting
                    time.sleep(0.1)
                else:
                    # Since models are sorted by downloads descending, we can break early
                    if hasattr(model, 'downloads') and model.downloads and model.downloads < self.download_threshold:
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            
        logger.info(f"Found {len(all_models)} models above threshold")
        return all_models
    
    def extract_hash_from_lfs_pointer(self, repo_id: str, filename: str) -> Optional[Tuple[str, int]]:
        """Extract SHA256 hash from Git LFS pointer file"""
        try:
            # Get the raw pointer file content with authentication
            url = f"https://huggingface.co/{repo_id}/raw/main/{filename}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                if content.startswith("version https://git-lfs.github.com/spec/v1"):
                    # Parse LFS pointer
                    lines = content.strip().split('\n')
                    sha256_hash = None
                    file_size = None
                    
                    for line in lines:
                        if line.startswith('oid sha256:'):
                            sha256_hash = line.split(':', 1)[1]
                        elif line.startswith('size '):
                            file_size = int(line.split(' ', 1)[1])
                    
                    if sha256_hash and file_size:
                        return sha256_hash, file_size
                        
        except Exception as e:
            logger.debug(f"Could not extract LFS hash for {repo_id}/{filename}: {e}")
            
        return None
    
    def download_and_hash_file(self, repo_id: str, filename: str, expected_size: Optional[int] = None, 
                              ignore_size_limit: bool = False) -> Optional[Tuple[str, int]]:
        """Download file and compute SHA256 hash with size checking and cleanup"""
        downloaded_file = None
        try:
            # Check file size before download (unless ignoring limit for failed hash updates)
            if not ignore_size_limit and expected_size and expected_size > self.max_file_size_bytes:
                logger.info(f"Skipping {repo_id}/{filename} - size {expected_size/1024/1024/1024:.1f}GB exceeds limit")
                return None
            
            # If size unknown, get it from API
            if not expected_size:
                expected_size = self._get_file_size_from_api(repo_id, filename)
                if not ignore_size_limit and expected_size and expected_size > self.max_file_size_bytes:
                    logger.info(f"Skipping {repo_id}/{filename} - size {expected_size/1024/1024/1024:.1f}GB exceeds limit")
                    return None
            
            logger.debug(f"Downloading {repo_id}/{filename} for hashing")
            
            # Download the file with extended timeout and progress
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.download_dir,
                resume_download=True,  # Enable resume if interrupted
                force_download=False,  # Use cache if available
                local_files_only=False,
                token=self.token  # Use authentication token
            )
            
            # Compute hash with progress for large files
            sha256_hash = hashlib.sha256()
            file_size = 0
            
            with open(downloaded_file, 'rb') as f:
                # Use larger chunks for better performance
                chunk_size = 64 * 1024  # 64KB chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256_hash.update(chunk)
                    file_size += len(chunk)
            
            logger.debug(f"Successfully hashed {repo_id}/{filename} - {file_size/1024/1024:.1f}MB")
            return sha256_hash.hexdigest(), file_size
            
        except Exception as e:
            logger.debug(f"Could not download and hash {repo_id}/{filename}: {e}")
            return None
        finally:
            # Clean up downloaded file
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.remove(downloaded_file)
                    logger.debug(f"Cleaned up downloaded file: {downloaded_file}")
                except Exception as e:
                    logger.warning(f"Could not clean up {downloaded_file}: {e}")
    
    def get_file_hash(self, repo_id: str, filename: str, file_info: Dict, 
                     ignore_size_limit: bool = False) -> Optional[FileInfo]:
        """Get file hash using hybrid approach with size limits"""
        
        # Check file size first
        file_size = None
        if hasattr(file_info, 'lfs') and file_info.lfs and 'size' in file_info.lfs:
            file_size = file_info.lfs['size']
        elif hasattr(file_info, 'size') and file_info.size:
            file_size = file_info.size
        
        # If file is too large, mark as failed (unless ignoring size limit)
        if not ignore_size_limit and file_size and file_size > self.max_file_size_bytes:
            return FileInfo(
                hash='FAILED_TOO_LARGE',
                filename=filename,
                repo_id=repo_id,
                file_size=file_size,
                file_type=Path(filename).suffix.lower(),
                last_modified=getattr(file_info, 'lastModified', '').isoformat() if hasattr(file_info, 'lastModified') and file_info.lastModified else '',
                download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                hash_method='FAILED_TOO_LARGE',
                commit_hash=''
            )
        
        # Method 1: Try to extract from file metadata if available
        lfs_info = getattr(file_info, 'lfs', None) if hasattr(file_info, 'lfs') else None
        if lfs_info and 'sha256' in lfs_info:
            return FileInfo(
                hash=lfs_info['sha256'],
                filename=filename,
                repo_id=repo_id,
                file_size=lfs_info.get('size', file_size or 0),
                file_type=Path(filename).suffix.lower(),
                last_modified=getattr(file_info, 'lastModified', '').isoformat() if hasattr(file_info, 'lastModified') and file_info.lastModified else '',
                download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                hash_method='api_metadata',
                commit_hash=''
            )
        
        # Method 2: Try to extract from LFS pointer file
        lfs_result = self.extract_hash_from_lfs_pointer(repo_id, filename)
        if lfs_result:
            sha256_hash, lfs_size = lfs_result
            
            # Check size limit again (unless ignoring)
            if not ignore_size_limit and lfs_size > self.max_file_size_bytes:
                return FileInfo(
                    hash='FAILED_TOO_LARGE',
                    filename=filename,
                    repo_id=repo_id,
                    file_size=lfs_size,
                    file_type=Path(filename).suffix.lower(),
                    last_modified=getattr(file_info, 'lastModified', '').isoformat() if hasattr(file_info, 'lastModified') and file_info.lastModified else '',
                    download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                    hash_method='FAILED_TOO_LARGE',
                    commit_hash=''
                )
            
            return FileInfo(
                hash=sha256_hash,
                filename=filename,
                repo_id=repo_id,
                file_size=lfs_size,
                file_type=Path(filename).suffix.lower(),
                last_modified=getattr(file_info, 'lastModified', '').isoformat() if hasattr(file_info, 'lastModified') and file_info.lastModified else '',
                download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                hash_method='lfs_pointer',
                commit_hash=''
            )
        
        # Method 3: Download and compute hash (fallback)
        download_result = self.download_and_hash_file(repo_id, filename, file_size, ignore_size_limit)
        if download_result:
            sha256_hash, actual_size = download_result
            return FileInfo(
                hash=sha256_hash,
                filename=filename,
                repo_id=repo_id,
                file_size=actual_size,
                file_type=Path(filename).suffix.lower(),
                last_modified=getattr(file_info, 'lastModified', '').isoformat() if hasattr(file_info, 'lastModified') and file_info.lastModified else '',
                download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                hash_method='downloaded',
                commit_hash=''
            )
        
        # All methods failed
        return FileInfo(
            hash='FAILED_NO_HASH',
            filename=filename,
            repo_id=repo_id,
            file_size=file_size or 0,
            file_type=Path(filename).suffix.lower(),
            last_modified=getattr(file_info, 'lastModified', '').isoformat() if hasattr(file_info, 'lastModified') and file_info.lastModified else '',
            download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            hash_method='FAILED_NO_HASH',
            commit_hash=''
        )
    
    def scan_model_files(self, repo_id: str) -> List[FileInfo]:
        """Scan all target files in a model repository"""
        file_infos = []
        
        try:
            # Get model info and files
            info = model_info(repo_id, token=self.token)
            
            # Filter for target file extensions
            target_files = [
                f for f in info.siblings 
                if any(f.rfilename.lower().endswith(ext) for ext in TARGET_EXTENSIONS)
            ]
            
            logger.debug(f"Found {len(target_files)} target files in {repo_id}")
            
            for file_info in target_files:
                filename = file_info.rfilename
                
                # Get file hash using hybrid approach
                file_data = self.get_file_hash(repo_id, filename, file_info)
                
                if file_data:
                    # Update commit hash from model info
                    file_data.commit_hash = getattr(info, 'sha', '')
                    file_infos.append(file_data)
                    self.processed_files += 1
                    
                    # Write to CSV immediately
                    self._write_file_info_to_csv(file_data)
                    
                    # Log failures appropriately
                    if file_data.hash_method.startswith('FAILED'):
                        if file_data.hash_method == 'FAILED_TOO_LARGE':
                            self.log_error(repo_id, filename, 'file_too_large', f'File size {file_data.file_size/1024/1024/1024:.1f}GB exceeds limit')
                        else:
                            self.log_error(repo_id, filename, 'hash_extraction_failed', 'Could not extract hash using any method')
                else:
                    self.log_error(repo_id, filename, 'processing_failed', 'Could not process file')
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
        except RepositoryNotFoundError:
            self.log_error(repo_id, '', 'repository_not_found', 'Repository does not exist or is private')
        except GatedRepoError:
            self.log_error(repo_id, '', 'gated_repository', 'Repository requires access request')
        except Exception as e:
            self.log_error(repo_id, '', 'model_info_error', str(e))
        
        return file_infos
    
    def _write_file_info_to_csv(self, file_info: FileInfo):
        """Write file info to CSV files immediately"""
        if self.basic_writer:
            self.basic_writer.writerow(file_info.to_basic_dict())
            self.basic_file.flush()
        
        if self.extended_writer:
            self.extended_writer.writerow(file_info.to_extended_dict())
            self.extended_file.flush()
    
    def _setup_csv_writers(self, output_dir: str, restart_files: Optional[Dict[str, str]] = None):
        """Setup CSV writers for streaming output"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if restart_files:
            # Restarting - use existing files
            basic_filename = restart_files['basic']
            extended_filename = restart_files['extended']
            errors_filename = restart_files['errors']
            
            # Open in append mode
            self.basic_file = open(basic_filename, 'a', newline='', encoding='utf-8')
            self.basic_writer = csv.DictWriter(self.basic_file, fieldnames=['hash', 'filename', 'repo_id'])
            
            self.extended_file = open(extended_filename, 'a', newline='', encoding='utf-8')
            fieldnames = list(FileInfo.__annotations__.keys())
            self.extended_writer = csv.DictWriter(self.extended_file, fieldnames=fieldnames)
            
            self.error_file = open(errors_filename, 'a', newline='', encoding='utf-8')
            error_fieldnames = ['repo_id', 'filename', 'error_type', 'reason', 'timestamp']
            self.error_writer = csv.DictWriter(self.error_file, fieldnames=error_fieldnames)
            
            logger.info(f"Restarting with existing CSV files:")
        else:
            # New scan - create new files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            basic_filename = output_path / f"hf_model_hashes_basic_{timestamp}.csv"
            extended_filename = output_path / f"hf_model_hashes_extended_{timestamp}.csv"
            errors_filename = output_path / f"hf_scan_errors_{timestamp}.csv"
            
            # Create new files with headers and scan parameters
            self._create_csv_with_parameters(basic_filename, ['hash', 'filename', 'repo_id'])
            self._create_csv_with_parameters(extended_filename, list(FileInfo.__annotations__.keys()))
            self._create_csv_with_parameters(errors_filename, ['repo_id', 'filename', 'error_type', 'reason', 'timestamp'], is_error_file=True)
            
            # Open for appending (headers already written)
            self.basic_file = open(str(basic_filename), 'a', newline='', encoding='utf-8')
            self.basic_writer = csv.DictWriter(self.basic_file, fieldnames=['hash', 'filename', 'repo_id'])
            
            self.extended_file = open(str(extended_filename), 'a', newline='', encoding='utf-8')
            fieldnames = list(FileInfo.__annotations__.keys())
            self.extended_writer = csv.DictWriter(self.extended_file, fieldnames=fieldnames)
            
            self.error_file = open(str(errors_filename), 'a', newline='', encoding='utf-8')
            error_fieldnames = ['repo_id', 'filename', 'error_type', 'reason', 'timestamp']
            self.error_writer = csv.DictWriter(self.error_file, fieldnames=error_fieldnames)
            
            logger.info(f"CSV files created:")
        
        logger.info(f"  Basic: {basic_filename}")
        logger.info(f"  Extended: {extended_filename}")
        logger.info(f"  Errors: {errors_filename}")
    
    def _create_csv_with_parameters(self, filename: Union[str, Path], fieldnames: List[str], is_error_file: bool = False):
        """Create CSV file with scan parameters at the top"""
        with open(str(filename), 'w', newline='', encoding='utf-8') as f:
            # Write scan parameters as comments
            f.write(f"# SCAN_PARAMETERS\n")
            f.write(f"# scan_start: {datetime.now().isoformat()}\n")
            f.write(f"# download_threshold: {self.download_threshold}, ")
            f.write(f"max_file_size_gb: {self.max_file_size_bytes / (1024**3)}, ")
            f.write(f"rate_limit_delay: {self.rate_limit_delay}, ")
            f.write(f"max_workers: {self.max_workers}\n")
            f.write(f"# target_extensions: {','.join(sorted(TARGET_EXTENSIONS))}\n")
            f.write(f"# authenticated: {'yes' if self.token else 'no'}\n")
            f.write(f"# END_PARAMETERS\n")
            if is_error_file:
                f.write(f"# output_format: error log\n")
            else: 
                f.write(f"# output_format: {'basic' if len(fieldnames) == 3 else 'extended'}\n")
                
            # Write CSV header
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def _close_csv_writers(self):
        """Close CSV file handles"""
        if self.basic_file:
            self.basic_file.close()
        if self.extended_file:
            self.extended_file.close()
        if self.error_file:
            self.error_file.close()
    
    def scan_models(self, models: List[Dict], output_dir: str, restart_files: Optional[Dict[str, str]] = None, 
                    processed_repos: Optional[Set[str]] = None) -> List[FileInfo]:
        """Scan multiple models for file hashes with streaming output"""
        all_file_infos = []
        
        # Filter out already processed models if restarting
        if processed_repos:
            original_count = len(models)
            models = [m for m in models if m['repo_id'] not in processed_repos]
            logger.info(f"Restarting: skipping {original_count - len(models)} already processed models")
        
        # Setup CSV writers for streaming
        self._setup_csv_writers(output_dir, restart_files)
        
        try:
            logger.info(f"Starting scan of {len(models)} models")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit scanning tasks
                future_to_repo = {
                    executor.submit(self.scan_model_files, model['repo_id']): model['repo_id']
                    for model in models
                }
                
                # Process completed tasks
                for future in tqdm(as_completed(future_to_repo), total=len(models), desc="Scanning models"):
                    repo_id = future_to_repo[future]
                    try:
                        file_infos = future.result()
                        all_file_infos.extend(file_infos)
                        self.processed_models += 1
                    except Exception as e:
                        self.log_error(repo_id, '', 'scan_error', str(e))
            
            logger.info(f"Scan complete: {self.processed_models} models, {self.processed_files} files")
            
        finally:
            # Close CSV writers
            self._close_csv_writers()
            
            # Clean up download directory
            if os.path.exists(self.download_dir):
                try:
                    # Only remove our temporary files, not the directory itself
                    for root, dirs, files in os.walk(self.download_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass  # Ignore cleanup errors
                    logger.info(f"Cleaned up download directory: {self.download_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean up download directory: {e}")
        
        return all_file_infos
    
    def save_metadata(self, output_dir: str):
        """Save scan metadata"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metadata = {
            'scan_timestamp': datetime.now().isoformat(),
            'download_threshold': self.download_threshold,
            'max_file_size_gb': self.max_file_size_bytes / (1024**3),
            'total_models_scanned': self.processed_models,
            'total_files_processed': self.processed_files,
            'total_errors': len(self.errors),
            'target_extensions': list(TARGET_EXTENSIONS),
            'rate_limit_delay': self.rate_limit_delay,
            'max_workers': self.max_workers,
            'authenticated': bool(self.token)
        }
        
        metadata_file = output_path / f"scan_metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_file}")
    
    def update_failed_hashes(self, output_dir: str) -> None:
        """Update files with FAILED_NO_HASH or FAILED_TOO_LARGE with authentication"""
        # Find most recent CSV files
        restart_files = find_most_recent_files(output_dir)
        if not restart_files:
            logger.error("No CSV files found in output directory")
            return
        
        # Confirm with user
        print(f"\nFound most recent scan files:")
        print(f"  Basic: {restart_files['basic']}")
        print(f"  Extended: {restart_files['extended']}")
        
        response = input("\nProceed with updating failed hashes in these files? (y/N): ")
        if response.lower() != 'y':
            logger.info("Update cancelled by user")
            return
        
        # Load failed entries from extended CSV
        failed_entries = []
        extended_file = restart_files['extended']
        
        logger.info(f"Loading failed entries from {extended_file}")
        
        try:
            with open(extended_file, 'r', encoding='utf-8') as f:
                # Skip parameter lines
                for line in f:
                    if line.startswith('# END_PARAMETERS'):
                        break
                
                # Read CSV data
                reader = csv.DictReader(f)
                for row in reader:
                    if row['hash'] in ['FAILED_NO_HASH', 'FAILED_TOO_LARGE']:
                        failed_entries.append(row)
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return
        
        if not failed_entries:
            logger.info("No failed entries found to update")
            return
        
        logger.info(f"Found {len(failed_entries)} failed entries to retry")
        
        # Group by repo_id for efficient processing
        failed_by_repo = {}
        for entry in failed_entries:
            repo_id = entry['repo_id']
            if repo_id not in failed_by_repo:
                failed_by_repo[repo_id] = []
            failed_by_repo[repo_id].append(entry)
        
        # Process failed entries with authentication and size limit override
        updated_entries = []
        
        for repo_id, entries in tqdm(failed_by_repo.items(), desc="Retrying failed repositories"):
            try:
                # Get model info with authentication
                info = model_info(repo_id, token=self.token)
                
                for entry in entries:
                    filename = entry['filename']
                    
                    # Find the file info from the model
                    file_info = None
                    for sibling in info.siblings:
                        if sibling.rfilename == filename:
                            file_info = sibling
                            break
                    
                    if file_info:
                        # Retry with authentication and ignore size limits
                        logger.info(f"Retrying {repo_id}/{filename}")
                        new_file_data = self.get_file_hash(repo_id, filename, file_info, ignore_size_limit=True)
                        
                        if new_file_data and not new_file_data.hash.startswith('FAILED'):
                            # Update commit hash
                            new_file_data.commit_hash = getattr(info, 'sha', '')
                            updated_entries.append((entry, new_file_data))
                            logger.info(f"Successfully updated hash for {repo_id}/{filename}")
                        else:
                            logger.warning(f"Still failed to get hash for {repo_id}/{filename}")
                    else:
                        logger.warning(f"File not found in repository: {repo_id}/{filename}")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error processing repository {repo_id}: {e}")
        
        if not updated_entries:
            logger.info("No hashes were successfully updated")
            return
        
        logger.info(f"Successfully updated {len(updated_entries)} file hashes")
        
        # Update both CSV files
        self._update_csv_files(restart_files, updated_entries)
        
        logger.info("CSV files updated successfully")
    
    def _update_csv_files(self, restart_files: Dict[str, str], updated_entries: List[Tuple[Dict, FileInfo]]):
        """Update both basic and extended CSV files with new hash values"""
        
        # Update extended CSV
        self._update_single_csv_file(restart_files['extended'], updated_entries, 'extended')
        
        # Update basic CSV
        self._update_single_csv_file(restart_files['basic'], updated_entries, 'basic')
    
    def _update_single_csv_file(self, csv_file: str, updated_entries: List[Tuple[Dict, FileInfo]], file_type: str):
        """Update a single CSV file with new hash values"""
        
        # Read all lines from the file
        lines = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            return
        
        # Find the data start (after parameters and header)
        data_start_idx = 0
        header_line = None
        
        for i, line in enumerate(lines):
            if line.startswith('# END_PARAMETERS'):
                # Next line should be the header
                if i + 1 < len(lines):
                    header_line = lines[i + 1].strip()
                    data_start_idx = i + 2
                break
        
        if header_line is None:
            logger.error(f"Could not find header in {csv_file}")
            return
        
        # Parse existing data
        fieldnames = [field.strip() for field in header_line.split(',')]
        existing_data = []
        
        for line_idx in range(data_start_idx, len(lines)):
            line = lines[line_idx].strip()
            if line:
                # Parse CSV line
                values = []
                in_quotes = False
                current_value = ""
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        values.append(current_value.strip())
                        current_value = ""
                    else:
                        current_value += char
                
                if current_value:
                    values.append(current_value.strip())
                
                # Create row dict
                row = {}
                for i, field in enumerate(fieldnames):
                    if i < len(values):
                        row[field] = values[i].strip('"')
                    else:
                        row[field] = ""
                
                existing_data.append(row)
        
        # Create lookup for updates
        update_lookup = {}
        for old_entry, new_file_data in updated_entries:
            key = (old_entry['repo_id'], old_entry['filename'])
            if file_type == 'basic':
                update_lookup[key] = new_file_data.to_basic_dict()
            else:
                update_lookup[key] = new_file_data.to_extended_dict()
        
        # Update the data
        updated_count = 0
        for row in existing_data:
            key = (row['repo_id'], row['filename'])
            if key in update_lookup:
                # Update this row
                updated_row = update_lookup[key]
                for field, value in updated_row.items():
                    if field in row:
                        row[field] = str(value)
                updated_count += 1
        
        # Write the updated file
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                # Write parameter lines
                for line in lines:
                    if line.startswith('# END_PARAMETERS'):
                        f.write(line)
                        break
                    f.write(line)
                
                # Write CSV data
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_data)
            
            logger.info(f"Updated {updated_count} entries in {csv_file}")
            
        except Exception as e:
            logger.error(f"Error writing updated {csv_file}: {e}")

def find_most_recent_files(output_dir: str) -> Optional[Dict[str, str]]:
    """Find the most recent set of CSV files in output directory"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    # Find most recent extended file (they have all the data)
    extended_files = list(output_path.glob("hf_model_hashes_extended_*.csv"))
    if not extended_files:
        return None
    
    # Sort by modification time, get most recent
    extended_file = max(extended_files, key=lambda x: x.stat().st_mtime)
    timestamp = extended_file.name.replace("hf_model_hashes_extended_", "").replace(".csv", "")
    
    basic_file = output_path / f"hf_model_hashes_basic_{timestamp}.csv"
    errors_file = output_path / f"hf_scan_errors_{timestamp}.csv"
    
    # Verify all files exist
    if basic_file.exists() and errors_file.exists():
        return {
            'basic': str(basic_file),
            'extended': str(extended_file),
            'errors': str(errors_file),
            'timestamp': timestamp
        }
    
    return None

def parse_csv_parameters(csv_file: str) -> Dict[str, Union[str, int, float, set]]:
    """Parse scan parameters from CSV file header"""
    parameters = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('# END_PARAMETERS'):
                    break
                if line.startswith('# ') and ':' in line:
                    # Parse parameter lines like "# download_threshold: 1000"
                    line = line[2:].strip()  # Remove "# "
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert values to appropriate types
                        if key in ['download_threshold', 'max_workers']:
                            try:
                                parameters[key] = int(value)
                            except ValueError:
                                logger.warning(f"Could not parse {key} as int: {value}")
                        elif key in ['max_file_size_gb', 'rate_limit_delay']:
                            try:
                                parameters[key] = float(value)
                            except ValueError:
                                logger.warning(f"Could not parse {key} as float: {value}")
                        elif key == 'target_extensions':
                            parameters[key] = set(value.split(',')) if value else set()
                        else:
                            parameters[key] = value
    except Exception as e:
        logger.error(f"Error parsing CSV parameters: {e}")
    
    return parameters

def get_processed_repos_from_csv(csv_file: str) -> Set[str]:
    """Get set of already processed repo_ids from CSV file"""
    processed_repos = set()
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip parameter lines
            for line in f:
                if line.startswith('# END_PARAMETERS'):
                    break
            
            # Read CSV data
            reader = csv.DictReader(f)
            for row in reader:
                if 'repo_id' in row and row['repo_id']:
                    processed_repos.add(row['repo_id'])
    except Exception as e:
        logger.error(f"Error reading processed repos: {e}")
    
    return processed_repos

def find_restart_files(output_dir: str, pattern: str = None) -> Optional[Dict[str, str]]:
    """Find most recent CSV files for restart"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    # Find most recent basic file
    basic_files = list(output_path.glob("hf_model_hashes_basic_*.csv"))
    if not basic_files:
        return None
    
    # Sort by modification time, get most recent
    basic_file = max(basic_files, key=lambda x: x.stat().st_mtime)
    timestamp = basic_file.name.replace("hf_model_hashes_basic_", "").replace(".csv", "")
    
    extended_file = output_path / f"hf_model_hashes_extended_{timestamp}.csv"
    errors_file = output_path / f"hf_scan_errors_{timestamp}.csv"
    
    if extended_file.exists() and errors_file.exists():
        return {
            'basic': str(basic_file),
            'extended': str(extended_file),
            'errors': str(errors_file)
        }
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Scan HuggingFace models for file hashes")
    parser.add_argument('--download-threshold', type=int, default=10000, help='Minimum download count for models')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for results')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum concurrent workers')
    parser.add_argument('--rate-limit', type=float, default=1.5, help='Delay between requests (seconds)')
    parser.add_argument('--max-file-size', type=float, default=5.0, help='Maximum file size to download (GB)')
    parser.add_argument('--limit', type=int, help='Limit number of models to scan (for testing)')
    parser.add_argument('--restart', action='store_true', help='Restart interrupted scan from most recent files')
    parser.add_argument('--restart-from', type=str, help='Restart from specific CSV file')
    parser.add_argument('--token', type=str, help='HuggingFace API token for authenticated access')
    parser.add_argument('--update-failed-hashes', action='store_true', help='Update FAILED_NO_HASH and FAILED_TOO_LARGE entries with authentication')
    
    args = parser.parse_args()
    
    # Log the command line for reproducibility (but mask token)
    command_line = ' '.join(sys.argv)
    if args.token:
        command_line = command_line.replace(args.token, '[TOKEN_MASKED]')
    logger.info(f"Command line: {command_line}")
    
    # Validate arguments
    if args.restart and args.restart_from:
        logger.error("Cannot use both --restart and --restart-from simultaneously")
        sys.exit(1)
    
    if args.update_failed_hashes and not args.token:
        logger.error("--update-failed-hashes requires --token for authentication")
        sys.exit(1)
    
    # Handle update failed hashes mode
    if args.update_failed_hashes:
        scanner = HuggingFaceScanner(token=args.token)
        scanner.update_failed_hashes(args.output_dir)
        return
    
    # Handle restart mode
    restart_files = None
    processed_repos = None
    scanner_params = None
    
    if args.restart or args.restart_from:
        if args.restart_from:
            # Use specific file
            restart_csv = args.restart_from
            if not os.path.exists(restart_csv):
                logger.error(f"Restart file not found: {restart_csv}")
                sys.exit(1)
        else:
            # Find most recent files
            restart_files_dict = find_restart_files(args.output_dir)
            if not restart_files_dict:
                logger.error("No restart files found in output directory")
                sys.exit(1)
            restart_csv = restart_files_dict['extended']
            restart_files = restart_files_dict
        
        # Parse parameters from CSV
        scanner_params = parse_csv_parameters(restart_csv)
        if scanner_params:
            logger.info("Restart mode: using parameters from previous scan")
            logger.info(f"Previous scan parameters: {scanner_params}")
            
            # Override command line args with CSV parameters
            if 'download_threshold' in scanner_params:
                args.download_threshold = scanner_params['download_threshold']
            if 'max_file_size_gb' in scanner_params:
                args.max_file_size = scanner_params['max_file_size_gb']
            if 'rate_limit_delay' in scanner_params:
                args.rate_limit = scanner_params['rate_limit_delay']
            if 'max_workers' in scanner_params:
                args.max_workers = scanner_params['max_workers']
        
        # Get already processed repos
        processed_repos = get_processed_repos_from_csv(restart_csv)
        logger.info(f"Found {len(processed_repos)} already processed repositories")
        
        if not restart_files:
            # Construct file paths from restart_csv
            restart_path = Path(restart_csv)
            timestamp = restart_path.name.replace("hf_model_hashes_extended_", "").replace(".csv", "")
            restart_files = {
                'basic': str(restart_path.parent / f"hf_model_hashes_basic_{timestamp}.csv"),
                'extended': restart_csv,
                'errors': str(restart_path.parent / f"hf_scan_errors_{timestamp}.csv")
            }
            
            # Verify all restart files exist
            for file_type, file_path in restart_files.items():
                if not os.path.exists(file_path):
                    logger.error(f"Restart {file_type} file not found: {file_path}")
                    sys.exit(1)
    
    # Initialize scanner
    scanner = HuggingFaceScanner(
        download_threshold=args.download_threshold,
        max_workers=args.max_workers,
        rate_limit_delay=args.rate_limit,
        max_file_size_gb=args.max_file_size,
        token=args.token
    )
    
    try:
        # Get models above threshold
        models = scanner.get_models_above_threshold(limit=args.limit)
        
        if not models:
            logger.warning("No models found above threshold")
            return
        
        # Scan models for file hashes (with streaming output and optional restart)
        file_infos = scanner.scan_models(models, args.output_dir, restart_files, processed_repos)
        
        # Save metadata
        scanner.save_metadata(args.output_dir)
        
        logger.info("Scan completed successfully!")
        logger.info(f"Command line: {command_line}")
        
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        logger.info("To restart, use: --restart or --restart-from <csv_file>")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        logger.info("To restart, use: --restart or --restart-from <csv_file>")
        sys.exit(1)

if __name__ == "__main__":
    main()