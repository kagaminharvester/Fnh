#!/usr/bin/env python
"""
Comprehensive CLI mode testing script.
Tests all available modes with all option combinations.
"""

import subprocess
import json
import os
import sys
from datetime import datetime
import tempfile
import shutil

def create_dummy_video(path, duration=1, rate=10, size='640x480'):
    """Creates a dummy video file using ffmpeg."""
    try:
        subprocess.run(
            ['ffmpeg', '-f', 'lavfi', '-i', f'color=c=black:s={size}:r={rate}', '-t', str(duration), path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error creating dummy video: {e}")
        if hasattr(e, 'stderr'):
            print(e.stderr)
        return False

def get_available_modes():
    """Dynamically discover all available CLI modes, excluding examples."""
    try:
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        return discovery.get_supported_cli_modes()
    except Exception as e:
        print(f"Error discovering modes: {e}")
        return []

MODES = get_available_modes()
MODES_WITH_OD = ["OFFLINE_3_STAGE", "OFFLINE_3_STAGE_MIXED"]

def run_test(test_video_path, output_dir, mode, autotune=True, od_mode=None):
    """Run a single test with specified options."""
    cmd = [
        "python", "main.py",
        test_video_path,
        "--mode", mode,
        "--overwrite",
        "--no-copy"
    ]
    
    if not autotune:
        cmd.append("--no-autotune")
    
    if od_mode and mode in MODES_WITH_OD:
        cmd.extend(["--od-mode", od_mode])
    
    test_name = f"{mode}"
    if not autotune:
        test_name += "_no-autotune"
    if od_mode and mode in MODES_WITH_OD:
        test_name += f"_od-{od_mode}"
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # Increased timeout
            env={**os.environ, "FUNGEN_TESTING": "1", "FUNGEN_OUTPUT_DIR": output_dir}
        )
        
        error_occurred = result.returncode != 0
        success = not error_occurred

        return {
            'test_name': test_name,
            'mode': mode,
            'autotune': autotune,
            'od_mode': od_mode,
            'success': success,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            'test_name': test_name,
            'mode': mode,
            'autotune': autotune,
            'od_mode': od_mode,
            'success': False,
            'error': 'TIMEOUT'
        }
    except Exception as e:
        return {
            'test_name': test_name,
            'mode': mode,
            'autotune': autotune,
            'od_mode': od_mode,
            'success': False,
            'error': str(e)
        }

def main():
    """Run all tests and generate report."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_video = os.path.join(temp_dir, "test_video.mp4")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        if not create_dummy_video(test_video):
            print("Could not create dummy video. Aborting tests.")
            return 1

        results = []
        
        print(f"Starting comprehensive CLI testing at {datetime.now()}")
        print(f"Test video: {test_video}")
        print(f"Total modes to test: {len(set(MODES))}")
        
        tested_modes = set()
        for mode in MODES:
            if mode in tested_modes:
                continue
            tested_modes.add(mode)

            results.append(run_test(test_video, output_dir, mode, autotune=True))
            results.append(run_test(test_video, output_dir, mode, autotune=False))

            if mode in MODES_WITH_OD:
                results.append(run_test(test_video, output_dir, mode, autotune=True, od_mode="legacy"))
                results.append(run_test(test_video, output_dir, mode, autotune=False, od_mode="legacy"))
        
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        total_tests = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total_tests - successful
        
        print(f"\nTotal tests: {total_tests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        print("\n" + "-"*80)
        print(f"{'Mode':<30} {'Autotune':<10} {'OD':<8} {'Success':<8}")
        print("-"*80)

        for r in results:
            mode = r['mode'][:28]
            autotune = "Yes" if r['autotune'] else "No"
            od = r['od_mode'] or "-"
            success = "✓" if r['success'] else "✗"
            print(f"{mode:<30} {autotune:<10} {od:<8} {success:<8}")
            if not r['success']:
                print(f"  RETURN CODE: {r.get('return_code', 'N/A')}")
                if r.get('stderr'):
                    print(f"  STDERR: {r['stderr'][:200]}...")
                if r.get('error'):
                    print(f"  ERROR: {r.get('error')}")


        with open('cli_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to cli_test_results.json")

        return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())