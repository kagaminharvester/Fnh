#!/usr/bin/env python3
"""
Benchmark Harness
Author: FunGen AI System
Version: 1.0.0

Runs performance benchmarks on test video clips and outputs metrics.
"""

import sys
import argparse
import logging
import json
import csv
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks on video clips."""
    
    def __init__(self, 
                 fungen_cmd: str = 'python main.py',
                 timeout: int = 600,
                 verbose: bool = False):
        """
        Initialize benchmark runner.
        
        Args:
            fungen_cmd: Command to run FunGen
            timeout: Timeout per video in seconds
            verbose: Enable verbose output
        """
        self.fungen_cmd = fungen_cmd
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        
    def run_benchmark(self,
                     video_path: Path,
                     mode: str,
                     extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run benchmark on single video.
        
        Args:
            video_path: Path to video file
            mode: Processing mode
            extra_args: Additional CLI arguments
            
        Returns:
            Benchmark result dictionary
        """
        logger.info(f"Benchmarking: {video_path.name} (mode: {mode})")
        
        # Build command
        cmd = self.fungen_cmd.split() + [
            str(video_path),
            '--mode', mode,
            '--overwrite',
            '--no-copy',
            '--profile-run'
        ]
        
        if extra_args:
            cmd.extend(extra_args)
            
        # Run benchmark
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={'FUNGEN_TESTING': '1'}
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output
            metrics = self._parse_output(result.stdout, result.stderr)
            metrics['video'] = video_path.name
            metrics['mode'] = mode
            metrics['total_time'] = elapsed_time
            metrics['success'] = result.returncode == 0
            metrics['error'] = None if result.returncode == 0 else 'Non-zero exit code'
            
            # Try to load profile JSON if available
            profile_path = video_path.parent / f"{video_path.stem}_profile.json"
            if profile_path.exists():
                try:
                    with open(profile_path) as f:
                        profile_data = json.load(f)
                        metrics.update(profile_data)
                except Exception as e:
                    logger.warning(f"Failed to load profile: {e}")
                    
            return metrics
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            logger.error(f"Timeout after {elapsed_time:.1f}s")
            return {
                'video': video_path.name,
                'mode': mode,
                'total_time': elapsed_time,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error: {e}")
            return {
                'video': video_path.name,
                'mode': mode,
                'total_time': elapsed_time,
                'success': False,
                'error': str(e)
            }
            
    def _parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse metrics from command output."""
        metrics = {}
        
        # Look for FPS in output
        for line in stdout.split('\n'):
            if 'FPS' in line and '100.00%' in line:
                try:
                    # Extract FPS from progress line
                    parts = line.split('|')
                    for part in parts:
                        if 'FPS' in part:
                            fps_str = part.strip().split()[0]
                            metrics['fps'] = float(fps_str)
                            break
                except Exception:
                    pass
                    
        # Look for errors in stderr
        if 'ERROR' in stderr or 'Exception' in stderr:
            metrics['has_errors'] = True
        else:
            metrics['has_errors'] = False
            
        return metrics
        
    def run_batch(self,
                 video_dir: Path,
                 modes: List[str],
                 extra_args: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run benchmarks on all videos in directory.
        
        Args:
            video_dir: Directory containing test videos
            modes: List of modes to test
            extra_args: Additional CLI arguments
            
        Returns:
            List of benchmark results
        """
        # Find video files
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
        videos = []
        for ext in video_extensions:
            videos.extend(video_dir.glob(f'*{ext}'))
            
        if not videos:
            logger.warning(f"No videos found in {video_dir}")
            return []
            
        logger.info(f"Found {len(videos)} videos")
        logger.info(f"Testing {len(modes)} modes")
        logger.info(f"Total benchmarks: {len(videos) * len(modes)}")
        
        results = []
        for video in sorted(videos):
            for mode in modes:
                result = self.run_benchmark(video, mode, extra_args)
                results.append(result)
                self.results.append(result)
                
                # Log summary
                if result['success']:
                    fps = result.get('fps', 'N/A')
                    logger.info(f"  ✓ {mode}: {fps} FPS")
                else:
                    error = result.get('error', 'Unknown')
                    logger.error(f"  ✗ {mode}: {error}")
                    
        return results
        
    def save_results(self, output_path: Path, format: str = 'json'):
        """
        Save benchmark results to file.
        
        Args:
            output_path: Output file path
            format: Output format ('json' or 'csv')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
            
        elif format == 'csv':
            if not self.results:
                logger.warning("No results to save")
                return
                
            # Get all keys from results
            keys = set()
            for r in self.results:
                keys.update(r.keys())
            keys = sorted(keys)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            logger.info(f"Results saved to {output_path}")
            
        else:
            logger.error(f"Unknown format: {format}")
            
    def print_summary(self):
        """Print summary of benchmark results."""
        if not self.results:
            logger.info("No results to summarize")
            return
            
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r['success'])
        
        logger.info(f"Total benchmarks: {total}")
        logger.info(f"Successful: {success} ({100*success/total:.1f}%)")
        logger.info(f"Failed: {total-success} ({100*(total-success)/total:.1f}%)")
        
        # Group by mode
        by_mode = {}
        for r in self.results:
            mode = r['mode']
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(r)
            
        logger.info("\nBy Mode:")
        for mode, results in sorted(by_mode.items()):
            fps_values = [r.get('fps') for r in results if r.get('fps')]
            if fps_values:
                avg_fps = sum(fps_values) / len(fps_values)
                logger.info(f"  {mode}: {avg_fps:.1f} FPS (avg)")
            else:
                logger.info(f"  {mode}: No FPS data")
                
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run FunGen benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all videos with default modes
  python run_benchmark.py videos/
  
  # Benchmark with specific modes
  python run_benchmark.py videos/ --modes 3-stage stage2
  
  # Save results to CSV
  python run_benchmark.py videos/ --output results.csv --format csv
  
  # Add extra arguments
  python run_benchmark.py videos/ --extra-args "--tensorrt --batch-size 8"
        """
    )
    
    parser.add_argument('video_dir', type=Path,
                       help='Directory containing test videos')
    parser.add_argument('--modes', nargs='+', default=['3-stage', 'stage2'],
                       help='Modes to benchmark (default: 3-stage stage2)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output file path (default: benchmark_results_<timestamp>.json)')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per video in seconds (default: 600)')
    parser.add_argument('--extra-args', type=str, default=None,
                       help='Extra CLI arguments (quoted string)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Validate video directory
    if not args.video_dir.exists():
        logger.error(f"Directory not found: {args.video_dir}")
        return 1
        
    # Parse extra args
    extra_args = args.extra_args.split() if args.extra_args else None
    
    # Create runner
    runner = BenchmarkRunner(
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Run benchmarks
    logger.info("=" * 60)
    logger.info("Starting Benchmark Run")
    logger.info("=" * 60)
    
    runner.run_batch(args.video_dir, args.modes, extra_args)
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"benchmark_results_{timestamp}.{args.format}")
        
    runner.save_results(args.output, args.format)
    
    # Print summary
    runner.print_summary()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
