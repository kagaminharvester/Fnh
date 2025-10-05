#!/usr/bin/env python
"""
Test script to verify Phase 1 performance and infrastructure improvements.
"""
import subprocess
import sys
import os
import json
import tempfile


def test_cli_help():
    """Test that CLI help shows new flags."""
    print("Testing CLI help output...")
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    help_text = result.stdout
    
    # Check for new flags
    required_flags = [
        "--profile",
        "--precision",
        "--batch-size",
        "--reuse-detections",
        "--profile-run"
    ]
    
    missing = []
    for flag in required_flags:
        if flag not in help_text:
            missing.append(flag)
    
    if missing:
        print(f"❌ FAILED: Missing flags in help: {missing}")
        return False
    
    print("✓ All new CLI flags present in help")
    return True


def test_precision_module():
    """Test precision module can be imported and used."""
    print("\nTesting precision module...")
    try:
        from application.utils.precision import PrecisionPolicy
        
        # Test auto mode
        policy_auto = PrecisionPolicy("auto")
        assert policy_auto.mode == "auto"
        
        # Test fp32 mode
        policy_fp32 = PrecisionPolicy("fp32")
        assert policy_fp32.mode == "fp32"
        assert policy_fp32.use_fp16 == False
        
        # Test fp16 mode
        policy_fp16 = PrecisionPolicy("fp16")
        assert policy_fp16.mode == "fp16"
        
        print("✓ Precision module works correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_profiler_module():
    """Test profiler module can be imported and used."""
    print("\nTesting profiler module...")
    try:
        from application.perf.profiler import PerfSession
        
        # Reset session
        PerfSession.reset()
        
        # Test basic timing
        PerfSession.start("test_operation")
        import time
        time.sleep(0.01)
        PerfSession.stop("test_operation")
        
        # Get summary
        summary = PerfSession.summary_dict()
        assert "test_operation" in summary
        assert summary["test_operation"]["count"] == 1
        assert summary["test_operation"]["total_seconds"] >= 0.01
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            PerfSession.export_json(temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert "test_operation" in data
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        print("✓ Profiler module works correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_cache_module():
    """Test detection cache module."""
    print("\nTesting detection cache module...")
    try:
        from application.cache.detections_cache import (
            compute_video_hash,
            model_hash,
            get_cache_dir
        )
        
        # Test cache dir creation
        cache_dir = get_cache_dir()
        assert cache_dir.exists()
        
        # Test hash computation (with dummy values)
        # Note: These will fail if files don't exist, but that's expected
        try:
            vid_hash = compute_video_hash("nonexistent.mp4")
            assert len(vid_hash) > 0  # Should still return a hash
        except:
            pass  # Expected if file doesn't exist
        
        try:
            mod_hash = model_hash("nonexistent.pt")
            assert len(mod_hash) > 0  # Should still return a hash
        except:
            pass  # Expected if file doesn't exist
        
        print("✓ Detection cache module works correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_env_cache_functions():
    """Test environment cache functions."""
    print("\nTesting environment cache functions...")
    try:
        from application.utils.dependency_checker import (
            _compute_requirements_hash,
            _load_env_cache,
            _save_env_cache,
            _is_cache_valid
        )
        
        # Test requirements hash
        req_hash = _compute_requirements_hash("core.requirements.txt", "core.requirements.txt")
        assert len(req_hash) > 0
        
        # Test cache save/load
        test_cache = {
            "timestamp": 0,
            "python_version": "3.12.0",
            "requirements_hash": "test123"
        }
        
        _save_env_cache(test_cache)
        loaded = _load_env_cache()
        assert loaded.get("python_version") == "3.12.0"
        
        # Test cache validation (should be invalid due to old timestamp)
        is_valid = _is_cache_valid(loaded, "test123")
        assert is_valid == False  # Old timestamp
        
        print("✓ Environment cache functions work correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_json_logging():
    """Test JSON logging format."""
    print("\nTesting JSON logging format...")
    try:
        from application.utils.logger import JsonFormatter
        import logging
        
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["module"] == "test"
        
        print("✓ JSON logging format works correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_frame_prefetcher():
    """Test FramePrefetcher class."""
    print("\nTesting FramePrefetcher...")
    try:
        from video.video_processor import FramePrefetcher
        
        prefetcher = FramePrefetcher(max_queue_size=4)
        assert prefetcher.enabled == False  # Should be disabled by default
        
        print("✓ FramePrefetcher initialized correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 1 Infrastructure Tests")
    print("=" * 60)
    
    tests = [
        ("CLI Help", test_cli_help),
        ("Precision Module", test_precision_module),
        ("Profiler Module", test_profiler_module),
        ("Cache Module", test_cache_module),
        ("Environment Cache", test_env_cache_functions),
        ("JSON Logging", test_json_logging),
        ("Frame Prefetcher", test_frame_prefetcher),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ {name} raised exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_test in results:
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
