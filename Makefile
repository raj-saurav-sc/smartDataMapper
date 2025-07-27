# Smart Data Mapper - Test Automation Makefile
# Usage: make <target>

.PHONY: help test test-unit test-integration test-performance test-all
.PHONY: coverage benchmark samples quality report clean install-test-deps
.PHONY: test-ecommerce test-crm test-financial test-healthcare

# Default target
help:
	@echo "Smart Data Mapper Test Suite"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  test              - Run all tests (quick)"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-performance  - Run performance tests only"
	@echo "  test-all          - Run all tests with verbose output"
	@echo "  coverage          - Run tests with coverage reporting"
	@echo "  benchmark         - Run performance benchmarks"
	@echo "  samples           - Create sample datasets"
	@echo "  quality           - Run code quality checks"
	@echo "  report            - Generate HTML test report"
	@echo "  clean             - Clean test artifacts"
	@echo ""
	@echo "Dataset-specific tests:"
	@echo "  test-ecommerce    - Test with e-commerce dataset"
	@echo "  test-crm          - Test with CRM dataset"
	@echo "  test-financial    - Test with financial dataset"
	@echo "  test-healthcare   - Test with healthcare dataset"
	@echo ""
	@echo "Setup:"
	@echo "  install-test-deps - Install testing dependencies"

# Quick test run
test:
	@echo "🧪 Running Quick Test Suite..."
	@python run_tests.py --all

# Specific test categories
test-unit:
	@echo "🧪 Running Unit Tests..."
	@python run_tests.py --unit --verbose

test-integration:
	@echo "🔗 Running Integration Tests..."
	@python run_tests.py --integration --verbose

test-performance:
	@echo "🚀 Running Performance Tests..."
	@python run_tests.py --performance --verbose

test-all:
	@echo "🎯 Running Complete Test Suite..."
	@python run_tests.py --all --verbose

# Coverage reporting
coverage:
	@echo "📊 Running Tests with Coverage..."
	@python run_tests.py --all --coverage
	@echo "📄 Coverage report generated in test_reports/coverage/"

# Performance benchmarking
benchmark:
	@echo "⏱️  Running Performance Benchmarks..."
	@python run_tests.py --benchmark

# Sample dataset creation
samples:
	@echo "📊 Creating Sample Datasets..."
	@python run_tests.py --samples
	@echo "✅ Sample datasets created in test_samples/"

# Code quality checks
quality:
	@echo "🔍 Running Code Quality Checks..."
	@python run_tests.py --quality

# Generate comprehensive report
report:
	@echo "📋 Generating Comprehensive Test Report..."
	@python run_tests.py --report
	@echo "📄 Report available in test_reports/test_summary.html"

# Dataset-specific tests
test-ecommerce: samples
	@echo "🛒 Testing with E-commerce Dataset..."
	@python run_tests.py --test-dataset ecommerce --verbose

test-crm: samples
	@echo "👥 Testing with CRM Dataset..."
	@python run_tests.py --test-dataset crm --verbose

test-financial: samples
	@echo "💰 Testing with Financial Dataset..."
	@python run_tests.py --test-dataset financial --verbose

test-healthcare: samples
	@echo "🏥 Testing with Healthcare Dataset..."
	@python run_tests.py --test-dataset healthcare --verbose

# Clean up test artifacts
clean:
	@echo "🧹 Cleaning test artifacts..."
	@rm -rf test_reports/
	@rm -rf test_samples/
	@rm -rf .pytest_cache/
	@rm -rf __pycache__/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "✅ Cleanup complete"

# Install testing dependencies
install-test-deps:
	@echo "📦 Installing testing dependencies..."
	@pip install -r test-requirements.txt
	@echo "✅ Testing dependencies installed"

# Continuous integration simulation
ci:
	@echo "🚀 Running CI Pipeline..."
	@make clean
	@make install-test-deps
	@make quality
	@make coverage
	@make benchmark
	@echo "✅ CI Pipeline completed successfully"

# Quick development test cycle
dev-test:
	@echo "🔄 Running Development Test Cycle..."
	@python run_tests.py --unit
	@make quality
	@echo "✅ Development tests passed"

# Full validation (like pre-commit)
validate:
	@echo "✅ Running Full Validation..."
	@make quality
	@make test-all
	@make benchmark
	@echo "🎉 All validations passed!"

# Help for individual commands
help-coverage:
	@echo "Coverage Testing:"
	@echo "  Runs all tests and generates coverage reports"
	@echo "  Output: test_reports/coverage/index.html"
	@echo "  Command: make coverage"

help-benchmark:
	@echo "Performance Benchmarking:"
	@echo "  Tests mapper performance with different schema sizes"
	@echo "  Measures execution time and memory usage"
	@echo "  Command: make benchmark"

help-quality:
	@echo "Code Quality Checks:"
	@echo "  - flake8: Linting and style checks"
	@echo "  - black: Code formatting verification"
	@echo "  - isort: Import statement organization"
	@echo "  Command: make quality"