# Smart Data Mapper - Testing Guide

This comprehensive testing guide covers all aspects of testing the Smart Data Mapper, from unit tests to performance benchmarks.

## 🎯 Test Suite Overview

The test suite is designed to ensure reliability, performance, and accuracy of the Smart Data Mapper across various scenarios and data types.

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows end-to-end
- **Performance Tests**: Measure execution time and memory usage
- **Regression Tests**: Ensure consistent behavior across versions
- **Real-world Dataset Tests**: Validate with industry-specific data

## 🚀 Quick Start

### Prerequisites

```bash
# Install testing dependencies
pip install -r test-requirements.txt

# Or using make
make install-test-deps
```

### Run All Tests

```bash
# Simple test run
python run_tests.py --all

# Or using make
make test

# With verbose output
make test-all
```

## 📋 Test Categories in Detail

### 1. Unit Tests (`TestAutoLearningMapper`)

Tests core functionality of the mapping algorithm:

- Field name preprocessing
- Abbreviation discovery
- String similarity calculations
- Data type inference
- Semantic grouping
- Pattern learning

```bash
# Run unit tests only
python run_tests.py --unit
make test-unit
```

**Key Test Cases:**
- `test_preprocess_field_name()`: Field name cleaning
- `test_abbreviation_discovery()`: Auto-detect abbreviations
- `test_enhanced_string_similarity()`: String matching algorithms
- `test_data_type_inference()`: Type detection from data
- `test_smart_suggest_mappings()`: Main mapping algorithm

### 2. Integration Tests (`TestIntegration`)

Tests complete workflows with realistic data:

- End-to-end mapping pipeline
- File loading and processing
- Report generation
- ETL script creation

```bash
# Run integration tests
python run_tests.py --integration
make test-integration
```

**Key Test Cases:**
- `test_end_to_end_pipeline()`: Complete mapping workflow
- `test_edge_cases()`: Handle empty/malformed data
- `test_performance_with_large_schemas()`: Scalability testing

### 3. File Operations (`TestFileOperations`)

Tests data loading and saving:

- CSV file handling with various delimiters
- JSON file processing (flat and nested)
- XML file support
- Mapping result persistence

```bash
# File operations are tested as part of unit tests
make test-unit
```

### 4. Real-world Dataset Tests (`TestSampleDatasets`)

Tests with industry-specific datasets:

- E-commerce data (products, orders, customers)
- CRM data (contacts, leads, companies)
- Financial data (accounts, transactions, balances)
- Healthcare data (patients, diagnoses, treatments)

```bash
# Test specific datasets
make test-ecommerce
make test-crm
make test-financial
make test-healthcare
```

## 🔧 Advanced Testing

### Coverage Testing

Generate detailed coverage reports:

```bash
# Run with coverage
python run_tests.py --all --coverage
make coverage

# View coverage report
open test_reports/coverage/index.html
```

**Coverage Targets:**
- Minimum 80% line coverage
- 90%+ coverage for core mapping logic
- 100% coverage for critical path functions

### Performance Benchmarking

Measure performance with different schema sizes:

```bash
# Run benchmarks
python run_tests.py --benchmark
make benchmark
```

**Benchmark Metrics:**
- Execution time vs. schema size
- Memory usage patterns
- Confidence score stability
- Mapping completeness rates

### Code Quality Checks

Ensure code standards and style:

```bash
# Run quality checks
python run_tests.py --quality
make quality
```

**Quality Tools:**
- **flake8**: Linting and PEP 8 compliance
- **black**: Code formatting consistency
- **isort**: Import statement organization

## 📊 Sample Datasets

The test suite includes realistic sample datasets for comprehensive testing:

### Creating Sample Datasets

```bash
# Create all sample datasets
python run_tests.py --samples
make samples

# Datasets created in test_samples/:
# - ecommerce_source.csv / ecommerce_target.csv
# - crm_source.csv / crm_target.csv
# - financial_source.csv / financial_target.csv
# - healthcare_source.csv / healthcare_target.csv
```

### Dataset Characteristics

#### E-commerce Dataset
- **Source**: Legacy product catalog (prod_id, product_nm, price_amt, cat_cd, inv_qty, created_ts)
- **Target**: Modern e-commerce schema (product_id, name, price, category, stock_quantity, date_added)
- **Expected Mappings**: 5-6 high-confidence mappings

#### CRM Dataset
- **Source**: Legacy contact system (contact_id, fname, lname, email_addr, phone_num, company, lead_score)
- **Target**: Modern CRM (id, full_name, email, contact_phone, organization, score)
- **Expected Mappings**: 5-6 mappings with name concatenation challenges

#### Financial Dataset
- **Source**: Legacy banking (account_num, cust_name, balance_amt, acct_type, open_date, branch_cd)
- **Target**: Modern banking (account_number, customer_name, current_balance, account_type, opening_date, branch_code)
- **Expected Mappings**: 6-7 high-confidence mappings

#### Healthcare Dataset
- **Source**: Legacy patient records (patient_id, pt_fname, pt_lname, dob, diagnosis_cd, admit_dt)
- **Target**: Modern EHR (patient_number, first_name, last_name, birth_date, primary_diagnosis, admission_date)
- **Expected Mappings**: 6-8 mappings with medical terminology

## 🎨 Test Reports

### HTML Test Report

Generate comprehensive HTML reports:

```bash
# Generate detailed report
python run_tests.py --report
make report

# View report
open test_reports/test_summary.html
```

**Report Includes:**
- Test execution summary
- Pass/fail status for each test
- Execution times
- Coverage metrics
- Failed test details

### Manual Testing with Sample Data

Test the mapper interactively:

```bash
# Test with e-commerce data
smart-mapper \
  --source test_samples/ecommerce_source.csv \
  --target test_samples/ecommerce_target.csv \
  --html-report ecommerce_report.html \
  --output ecommerce_mappings.json \
  --generate-script ecommerce_transform.py

# Test with CRM data
smart-mapper \
  --source test_samples/crm_source.csv \
  --target test_samples/crm_target.csv \
  --html-report crm_report.html \
  --min-confidence 0.3
```

## 🔄 Continuous Integration

### CI Pipeline Simulation

```bash
# Run complete CI pipeline
make ci
```

**CI Steps:**
1. Clean previous artifacts
2. Install dependencies
3. Run code quality checks
4. Execute full test suite with coverage
5. Run performance benchmarks
6. Generate reports

### Pre-commit Validation

```bash
# Quick development cycle
make dev-test

# Full validation (before commits)
make validate
```

## 🐛 Debugging Tests

### Running Specific Tests

```bash
# Run single test class
python -m pytest test_smart_mapper.py::TestAutoLearningMapper -v

# Run single test method
python -m pytest test_smart_mapper.py::TestAutoLearningMapper::test_abbreviation_discovery -v

# Run with debugger
python -m pytest test_smart_mapper.py::TestAutoLearningMapper::test_smart_suggest_mappings -v --pdb
```

### Verbose Output

```bash
# Maximum verbosity
python run_tests.py --all --verbose

# With test details
python -m pytest test_smart_mapper.py -vvv --tb=long
```

### Test Data Inspection

```bash
# Create samples and inspect
make samples
ls -la test_samples/
head test_samples/ecommerce_source.csv
```

## 📈 Performance Expectations

### Benchmark Targets

| Schema Size | Max Time | Max Memory | Min Accuracy |
|-------------|----------|------------|--------------|
| 10 fields   | < 2s     | < 100MB    | 90%          |
| 25 fields   | < 5s     | < 200MB    | 85%          |
| 50 fields   | < 15s    | < 400MB    | 80%          |
| 100 fields  | < 30s    | < 800MB    | 75%          |

### Quality Metrics

- **High Confidence Mappings** (≥0.7): Should find 60%+ of obvious matches
- **Medium Confidence Mappings** (0.5-0.7): Should capture semantic similarities
- **Low False Positives**: <5% of high-confidence mappings should be incorrect
- **Consistency**: Same inputs should produce identical results across runs

## 🛠️ Extending Tests

### Adding New Test Cases

1. **Create test method** in appropriate test class
2. **Follow naming convention**: `test_descriptive_name()`
3. **Use appropriate assertions**: `assertEqual`, `assertGreater`, etc.
4. **Add docstrings** explaining test purpose
5. **Include edge cases** and error conditions

### Adding New Sample Datasets

1. **Edit `create_sample_datasets()`** in `test_smart_mapper.py`
2. **Add realistic field names** and data patterns
3. **Include challenging mapping scenarios**
4. **Update documentation** with expected results

### Custom Test Scenarios

```python
def test_custom_scenario(self):
    """Test mapper with custom business scenario"""
    source_fields = ['legacy_field1', 'old_field2']
    target_fields = ['modern_field1', 'new_field2']
    
    # Create test data
    source_df = pd.DataFrame({...})
    target_df = pd.DataFrame({...})
    
    # Run mapping
    mappings = self.mapper.smart_suggest_mappings(
        source_fields, target_fields, source_df, target_df
    )
    
    # Assert expectations
    self.assertGreater(len(mappings), 0)
    # Add specific assertions for your scenario
```

## 📚 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Run `make samples` to create test datasets
3. **Performance Issues**: Check available memory and CPU
4. **Coverage Issues**: Ensure all code paths are tested

### Debug Commands

```bash
# Check dependencies
pip list | grep -E "(pandas|scikit|sentence)"

# Verify sample data
python -c "import pandas as pd; print(pd.read_csv('test_samples/ecommerce_source.csv').head())"

# Test individual components
python -c "from smartautoMapper import AutoLearningMapper; m = AutoLearningMapper(); print('OK')"
```

## 🎉 Best Practices

1. **Run tests frequently** during development
2. **Add tests for new features** immediately
3. **Test edge cases** and error conditions
4. **Keep tests independent** and isolated
5. **Use descriptive test names** and documentation
6. **Monitor performance** with regular benchmarks
7. **Maintain high coverage** for critical paths
8. **Update sample data** to reflect real-world complexity

---

## 📞 Support

For testing issues or questions:

1. Check this documentation first
2. Run `make help` for available commands
3. Examine test output for specific error details
4. Create sample datasets to reproduce issues
5. Report bugs with minimal reproduction cases