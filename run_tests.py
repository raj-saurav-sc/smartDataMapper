#!/usr/bin/env python3
"""
Comprehensive test runner for Smart Data Mapper
Provides various testing modes and reporting options
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
import json


class TestRunner:
    """Test runner with multiple modes and reporting"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def run_unit_tests(self, verbose=False):
        """Run unit tests only"""
        print("🧪 Running Unit Tests...")
        cmd = [
            "/bin/python3.11", "-m", "pytest", 
            "test_smart_mapper.py::TestAutoLearningMapper",
            "test_smart_mapper.py::TestFileOperations",
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests only"""
        print("🔗 Running Integration Tests...")
        cmd = [
            "/bin/python3.11", "-m", "pytest",
            "test_smart_mapper.py::TestIntegration",
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_performance_tests(self, verbose=False):
        """Run performance and benchmark tests"""
        print("🚀 Running Performance Tests...")
        cmd = [
            "/bin/python3.11", "-m", "pytest",
            "test_smart_mapper.py::TestRegressionSuite",
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_all_tests(self, verbose=False, coverage=False):
        """Run all tests with optional coverage"""
        print("🎯 Running Full Test Suite...")
        
        cmd = ["/bin/python3.11", "-m", "pytest", "test_smart_mapper.py"]
        
        if coverage:
            cmd.extend([
                "--cov=smartautoMapper",
                "--cov=report_generator", 
                "--cov=etl_generator",
                "--cov=visualize_mappings",
                "--cov-report=html:" + str(self.reports_dir / "coverage"),
                "--cov-report=term-missing"
            ])
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
            
        cmd.extend(["--tb=short", "--color=yes"])
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def create_sample_datasets(self, output_dir="test_samples"):
        """Create sample datasets for testing"""
        print(f"📊 Creating sample datasets in {output_dir}...")
        cmd = [
            "python", "test_smart_mapper.py", 
            "--create-samples", 
            "--sample-dir", output_dir
        ]
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_benchmarks(self):
        """Run performance benchmarks"""
        print("⏱️  Running Performance Benchmarks...")
        cmd = ["python", "test_smart_mapper.py", "--benchmark"]
        return subprocess.run(cmd, cwd=self.project_root)
    
    def test_with_sample_data(self, dataset_name="ecommerce", verbose=False):
        """Test the mapper with generated sample data"""
        sample_dir = self.project_root / "test_samples"
        
        if not sample_dir.exists():
            print("📊 Sample datasets not found. Creating them...")
            self.create_sample_datasets()
        
        print(f"🧪 Testing mapper with {dataset_name} dataset...")
        
        source_file = sample_dir / f"{dataset_name}_source.csv"
        target_file = sample_dir / f"{dataset_name}_target.csv"
        output_file = self.reports_dir / f"{dataset_name}_mappings.json"
        html_report = self.reports_dir / f"{dataset_name}_report.html"
        
        if not source_file.exists() or not target_file.exists():
            print(f"❌ {dataset_name} dataset files not found!")
            return False
        
        # Run the mapper
        cmd = [
            "python", "-m", "smartautoMapper",
            "--source", str(source_file),
            "--target", str(target_file),
            "--output", str(output_file),
            "--html-report", str(html_report),
            "--min-confidence", "0.2"
        ]
        
        if verbose:
            print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print(f"✅ Test completed successfully!")
            print(f"📄 Results saved to: {output_file}")
            print(f"🌐 HTML report: {html_report}")
        else:
            print(f"❌ Test failed with return code {result.returncode}")
        
        return result.returncode == 0
    
    def run_code_quality_checks(self):
        """Run code quality checks"""
        print("🔍 Running Code Quality Checks...")
        
        # Check if tools are available
        tools = ['flake8', 'black', 'isort']
        missing_tools = []
        
        for tool in tools:
            try:
                subprocess.run([tool, '--version'], 
                             capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"⚠️  Missing tools: {', '.join(missing_tools)}")
            print("Install with: pip install flake8 black isort")
            return False
        
        # Run flake8
        print("  → Running flake8 (linting)...")
        flake8_result = subprocess.run([
            'flake8', '--max-line-length=100', '--ignore=E203,W503',
            '*.py'
        ], cwd=self.project_root)
        
        # Run black check
        print("  → Running black (formatting check)...")
        black_result = subprocess.run([
            'black', '--check', '--line-length=100', '*.py'
        ], cwd=self.project_root)
        
        # Run isort check
        print("  → Running isort (import sorting check)...")
        isort_result = subprocess.run([
            'isort', '--check-only', '--line-length=100', '*.py'
        ], cwd=self.project_root)
        
        all_passed = all(r.returncode == 0 for r in [flake8_result, black_result, isort_result])
        
        if all_passed:
            print("✅ All code quality checks passed!")
        else:
            print("❌ Some code quality checks failed")
            
        return all_passed
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("📋 Generating Comprehensive Test Report...")
        
        report_file = self.reports_dir / "test_summary.html"
        
        cmd = [
            "python", "-m", "pytest", "test_smart_mapper.py",
            "--html=" + str(report_file),
            "--self-contained-html",
            "--cov=smartautoMapper",
            "--cov-report=html:" + str(self.reports_dir / "coverage"),
            "-v"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print(f"✅ Test report generated: {report_file}")
        
        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Smart Data Mapper Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --verbose         # Run unit tests with verbose output
  python run_tests.py --integration            # Run integration tests only
  python run_tests.py --performance            # Run performance tests
  python run_tests.py --samples                # Create sample datasets
  python run_tests.py --test-dataset ecommerce # Test with ecommerce dataset
  python run_tests.py --benchmark              # Run performance benchmarks
  python run_tests.py --quality                # Run code quality checks
  python run_tests.py --report                 # Generate HTML test report
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--all', action='store_true',
                           help='Run all tests (default)')
    test_group.add_argument('--unit', action='store_true',
                           help='Run unit tests only')
    test_group.add_argument('--integration', action='store_true',
                           help='Run integration tests only')
    test_group.add_argument('--performance', action='store_true',
                           help='Run performance tests only')
    
    # Utility options
    parser.add_argument('--samples', action='store_true',
                       help='Create sample datasets for testing')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--test-dataset', choices=['ecommerce', 'crm', 'financial', 'healthcare'],
                       help='Test mapper with specific sample dataset')
    parser.add_argument('--quality', action='store_true',
                       help='Run code quality checks (flake8, black, isort)')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive HTML test report')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    start_time = time.time()
    
    try:
        # Handle utility commands first
        if args.samples:
            result = runner.create_sample_datasets()
            return result.returncode
        
        if args.benchmark:
            result = runner.run_benchmarks()
            return result.returncode
        
        if args.test_dataset:
            success = runner.test_with_sample_data(args.test_dataset, args.verbose)
            return 0 if success else 1
        
        if args.quality:
            success = runner.run_code_quality_checks()
            return 0 if success else 1
        
        if args.report:
            success = runner.generate_test_report()
            return 0 if success else 1
        
        # Run tests based on selection
        if args.unit:
            result = runner.run_unit_tests(args.verbose)
        elif args.integration:
            result = runner.run_integration_tests(args.verbose)
        elif args.performance:
            result = runner.run_performance_tests(args.verbose)
        else:  # default to all tests
            result = runner.run_all_tests(args.verbose, args.coverage)
        
        end_time = time.time()
        print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())