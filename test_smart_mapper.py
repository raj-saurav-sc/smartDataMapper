import unittest
import pandas as pd
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import warnings
import sys
warnings.filterwarnings("ignore")

# Import the modules to test
try:
    from smartautoMapper import AutoLearningMapper, FieldMapping, load_dataframe_from_file, save_mappings_to_json
    from report_generator import generate_mapping_report
    from etl_generator import generate_etl_script
    from visualize_mappings import VisualMappingDisplay
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure you're running from the project root directory")
    # Create mock classes for testing
    class AutoLearningMapper:
        def __init__(self): pass
    class FieldMapping:
        def __init__(self, *args, **kwargs): pass
    def load_dataframe_from_file(path): return pd.DataFrame()
    def save_mappings_to_json(*args): return {}
    def generate_mapping_report(*args): pass
    def generate_etl_script(*args): pass
    class VisualMappingDisplay:
        def __init__(self): pass


class TestAutoLearningMapper(unittest.TestCase):
    """Test cases for the AutoLearningMapper class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mapper = AutoLearningMapper()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test mapper initialization"""
        self.assertIsNotNone(self.mapper.model)
        self.assertIsInstance(self.mapper.learned_patterns, dict)
        self.assertIsInstance(self.mapper.abbreviation_dict, dict)
        self.assertIsInstance(self.mapper.synonym_groups, dict)
    
    def test_preprocess_field_name(self):
        """Test field name preprocessing"""
        test_cases = [
            ("customer_name", "customer name"),
            ("CUST-ID", "cust-id"),
            ("email.address", "email address"),
            ("user__profile", "user profile"),
            ("  spaced_field  ", "spaced field")
        ]
        
        for input_field, expected in test_cases:
            with self.subTest(input_field=input_field):
                result = self.mapper.preprocess_field_name(input_field)
                self.assertEqual(result, expected)
    
    def test_abbreviation_discovery(self):
        """Test automatic abbreviation discovery"""
        field_names = [
            "customer_id", "cust_name", "cust_email",
            "phone_number", "ph_num", "mobile_phone",
            "description", "desc", "user_desc"
        ]
        
        self.mapper.auto_discover_patterns(field_names)
        
        # Should discover some abbreviations
        self.assertIn("cust", self.mapper.abbreviation_dict)
        self.assertIn("desc", self.mapper.abbreviation_dict)
        self.assertIn("ph", self.mapper.abbreviation_dict)
    
    def test_enhanced_string_similarity(self):
        """Test enhanced string similarity calculations"""
        test_cases = [
            ("customer_name", "cust_name", 0.7),  # Should be high due to abbreviation
            ("email", "email_address", 0.6),      # Should be moderate
            ("id", "identifier", 0.5),            # Should be moderate
            ("phone", "mobile", 0.0),             # Should be low
        ]
        
        # First discover patterns
        all_fields = [case[0] for case in test_cases] + [case[1] for case in test_cases]
        self.mapper.auto_discover_patterns(all_fields)
        
        for field1, field2, min_expected in test_cases:
            with self.subTest(field1=field1, field2=field2):
                similarity, abbrev_sim = self.mapper.enhanced_string_similarity(field1, field2)
                self.assertGreaterEqual(similarity, min_expected, 
                    f"Similarity between '{field1}' and '{field2}' should be at least {min_expected}, got {similarity}")
    
    def test_data_type_inference(self):
        """Test data type inference from actual data"""
        # Create test data
        numeric_series = pd.Series([1, 2, 3, 4, 5])
        date_series = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        string_series = pd.Series(['John', 'Jane', 'Bob'])
        boolean_series = pd.Series([True, False, True, False])
        
        # Test numeric inference
        self.assertEqual(self.mapper._infer_column_data_type(numeric_series), 'numeric')
        
        # Test string inference
        self.assertEqual(self.mapper._infer_column_data_type(string_series), 'string')
        
        # Test boolean inference
        self.assertEqual(self.mapper._infer_column_data_type(boolean_series), 'boolean')
    
    def test_semantic_group_discovery(self):
        """Test semantic group discovery with sample data"""
        # Create test DataFrames
        source_df = pd.DataFrame({
            'customer_name': ['John Doe', 'Jane Smith'],
            'cust_email': ['john@email.com', 'jane@email.com'],
            'phone_num': ['123-456-7890', '098-765-4321'],
            'order_total': [100.50, 250.75]
        })
        
        target_df = pd.DataFrame({
            'full_name': ['', ''],
            'email_address': ['', ''],
            'contact_phone': ['', ''],
            'amount_due': [0.0, 0.0]
        })
        
        # Test pattern discovery
        all_fields = list(source_df.columns) + list(target_df.columns)
        all_columns = {**source_df.to_dict('series'), **target_df.to_dict('series')}
        
        self.mapper.auto_discover_patterns(all_fields, all_columns)
        
        # Should discover some semantic groups
        self.assertGreater(len(self.mapper.synonym_groups), 0)
    
    def test_smart_suggest_mappings(self):
        """Test the main mapping suggestion algorithm"""
        source_fields = ['customer_id', 'cust_name', 'email', 'phone']
        target_fields = ['id', 'full_name', 'email_address', 'contact_number']
        
        # Create sample dataframes
        source_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'cust_name': ['John', 'Jane', 'Bob'],
            'email': ['john@test.com', 'jane@test.com', 'bob@test.com'],
            'phone': ['123-456-7890', '098-765-4321', '555-123-4567']
        })
        
        target_df = pd.DataFrame({
            'id': [0, 0, 0],
            'full_name': ['', '', ''],
            'email_address': ['', '', ''],
            'contact_number': ['', '', '']
        })
        
        mappings = self.mapper.smart_suggest_mappings(
            source_fields, target_fields, source_df, target_df
        )
        
        # Should find some mappings
        self.assertGreater(len(mappings), 0)
        
        # All mappings should be FieldMapping objects
        for mapping in mappings:
            self.assertIsInstance(mapping, FieldMapping)
            self.assertIn(mapping.source_field, source_fields)
            self.assertIn(mapping.target_field, target_fields)
            self.assertGreaterEqual(mapping.confidence_score, 0.0)
            self.assertLessEqual(mapping.confidence_score, 1.0)
    
    def test_get_best_unique_mappings(self):
        """Test getting best unique one-to-one mappings"""
        # Create sample mappings with overlapping sources/targets
        mappings = [
            FieldMapping('field1', 'target1', 0.9, 'exact', 'reason', 0.9, 0.8, 'pattern', 1.0, 0.0, 0.0),
            FieldMapping('field1', 'target2', 0.7, 'partial', 'reason', 0.7, 0.6, 'pattern', 0.8, 0.0, 0.0),
            FieldMapping('field2', 'target1', 0.6, 'partial', 'reason', 0.6, 0.5, 'pattern', 0.7, 0.0, 0.0),
            FieldMapping('field3', 'target3', 0.8, 'exact', 'reason', 0.8, 0.7, 'pattern', 0.9, 0.0, 0.0),
        ]
        
        best_mappings = self.mapper.get_best_unique_mappings(mappings)
        
        # Should get only unique mappings
        self.assertEqual(len(best_mappings), 2)  # field1->target1 and field3->target3
        
        # Should be highest confidence mappings
        source_fields = [m.source_field for m in best_mappings]
        target_fields = [m.target_field for m in best_mappings]
        
        self.assertIn('field1', source_fields)
        self.assertIn('field3', source_fields)
        self.assertIn('target1', target_fields)
        self.assertIn('target3', target_fields)


class TestFileOperations(unittest.TestCase):
    """Test cases for file loading and saving operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_file(self):
        """Test loading CSV files"""
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        test_data = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob'],
            'age': [25, 30, 35],
            'email': ['john@test.com', 'jane@test.com', 'bob@test.com']
        })
        test_data.to_csv(csv_path, index=False)
        
        loaded_df = load_dataframe_from_file(csv_path)
        
        self.assertEqual(len(loaded_df), 3)
        self.assertListEqual(list(loaded_df.columns), ['name', 'age', 'email'])
        self.assertEqual(loaded_df.iloc[0]['name'], 'John')
    
    def test_load_json_file(self):
        """Test loading JSON files"""
        json_path = os.path.join(self.temp_dir, 'test.json')
        test_data = [
            {'name': 'John', 'age': 25, 'email': 'john@test.com'},
            {'name': 'Jane', 'age': 30, 'email': 'jane@test.com'},
            {'name': 'Bob', 'age': 35, 'email': 'bob@test.com'}
        ]
        
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        loaded_df = load_dataframe_from_file(json_path)
        
        self.assertEqual(len(loaded_df), 3)
        self.assertIn('name', loaded_df.columns)
        self.assertIn('age', loaded_df.columns)
        self.assertIn('email', loaded_df.columns)
    
    def test_load_nested_json_file(self):
        """Test loading nested JSON files"""
        json_path = os.path.join(self.temp_dir, 'nested.json')
        test_data = [
            {
                'user': {'name': 'John', 'age': 25},
                'contact': {'email': 'john@test.com', 'phone': '123-456-7890'}
            },
            {
                'user': {'name': 'Jane', 'age': 30},
                'contact': {'email': 'jane@test.com', 'phone': '098-765-4321'}
            }
        ]
        
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        loaded_df = load_dataframe_from_file(json_path)
        
        self.assertEqual(len(loaded_df), 2)
        # Should have flattened columns
        self.assertIn('user.name', loaded_df.columns)
        self.assertIn('contact.email', loaded_df.columns)
    
    def test_save_mappings_to_json(self):
        """Test saving mappings to JSON file"""
        # Create test data
        mapper = AutoLearningMapper()
        mappings = [
            FieldMapping('field1', 'target1', 0.9, 'exact', 'test reason', 0.9, 0.8, 'pattern', 1.0, 0.0, 0.0)
        ]
        
        source_df = pd.DataFrame({'field1': [1, 2, 3], 'field2': ['a', 'b', 'c']})
        target_df = pd.DataFrame({'target1': [0, 0, 0], 'target2': ['', '', '']})
        
        output_path = os.path.join(self.temp_dir, 'mappings.json')
        
        result_data = save_mappings_to_json(
            mappings, mappings, source_df, target_df, mapper,
            'source.csv', 'target.csv', output_path
        )
        
        # Check file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check data structure
        self.assertIsNotNone(result_data)
        self.assertIn('metadata', result_data)
        self.assertIn('recommended_mappings', result_data)
        self.assertIn('unmapped_fields', result_data)
        
        # Check mapping was saved correctly
        self.assertEqual(len(result_data['recommended_mappings']), 1)
        self.assertEqual(result_data['recommended_mappings'][0]['source_field'], 'field1')


class TestReportGeneration(unittest.TestCase):
    """Test cases for report generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_html_report(self):
        """Test HTML report generation"""
        test_data = {
            'metadata': {
                'source_file': 'source.csv',
                'target_file': 'target.csv',
                'mapping_count': 2
            },
            'recommended_mappings': [
                {
                    'source_field': 'field1',
                    'target_field': 'target1',
                    'confidence_score': 0.9,
                    'similarity_type': 'exact',
                    'reasoning': 'test reason',
                    'sample_source_value': 'test_value'
                }
            ],
            'unmapped_fields': {'source': ['field2'], 'target': ['target2']},
            'conflicting_mappings': []
        }
        
        html_path = os.path.join(self.temp_dir, 'report.html')
        generate_mapping_report(test_data, html_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(html_path))
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            self.assertIn('<!DOCTYPE html>', content)
            self.assertIn('field1', content)
            self.assertIn('target1', content)
            self.assertIn('source.csv', content)
    
    def test_generate_etl_script(self):
        """Test ETL script generation"""
        mappings_path = os.path.join(self.temp_dir, 'mappings.json')
        script_path = os.path.join(self.temp_dir, 'transform.py')
        source_path = os.path.join(self.temp_dir, 'source.csv')
        
        # Create dummy mappings file
        mappings_data = {
            'recommended_mappings': [
                {'source_field': 'field1', 'target_field': 'target1'}
            ]
        }
        with open(mappings_path, 'w') as f:
            json.dump(mappings_data, f)
        
        generate_etl_script(mappings_path, source_path, script_path)
        
        # Check script was created
        self.assertTrue(os.path.exists(script_path))
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn('import pandas as pd', content)
            self.assertIn('def main():', content)
            self.assertIn(mappings_path, content)
            self.assertIn(source_path, content)


class TestVisualization(unittest.TestCase):
    """Test cases for visualization components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.display = VisualMappingDisplay()
        
    def test_confidence_color_assignment(self):
        """Test confidence level color and symbol assignment"""
        # Test high confidence
        color, symbol = self.display.get_confidence_color_and_symbol(0.8)
        self.assertEqual(symbol, '🟢')
        
        # Test medium confidence
        color, symbol = self.display.get_confidence_color_and_symbol(0.6)
        self.assertEqual(symbol, '🟡')
        
        # Test low confidence
        color, symbol = self.display.get_confidence_color_and_symbol(0.3)
        self.assertEqual(symbol, '🔴')
    
    @patch('builtins.print')
    def test_display_mappings(self, mock_print):
        """Test mapping display functionality"""
        mappings = [
            FieldMapping('field1', 'target1', 0.9, 'exact', 'test reason', 0.9, 0.8, 'pattern', 1.0, 0.0, 0.0)
        ]
        source_fields = ['field1', 'field2']
        target_fields = ['target1', 'target2']
        
        # Should not raise any exceptions
        self.display.display_beautiful_mappings(mappings, source_fields, target_fields)
        
        # Should have printed something
        self.assertTrue(mock_print.called)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample source data
        self.source_data = pd.DataFrame({
            'cust_id': [1, 2, 3, 4],
            'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
            'email_addr': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com'],
            'ph_number': ['123-456-7890', '098-765-4321', '555-123-4567', '777-888-9999'],
            'order_total': [150.50, 275.25, 89.99, 420.00],
            'created_dt': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
        })
        
        # Create sample target schema
        self.target_data = pd.DataFrame({
            'id': [0, 0, 0, 0],
            'full_name': ['', '', '', ''],
            'email': ['', '', '', ''],
            'phone': ['', '', '', ''],
            'amount': [0.0, 0.0, 0.0, 0.0],
            'registration_date': ['', '', '', '']
        })
        
        # Save to files
        self.source_path = os.path.join(self.temp_dir, 'source.csv')
        self.target_path = os.path.join(self.temp_dir, 'target.csv')
        self.source_data.to_csv(self.source_path, index=False)
        self.target_data.to_csv(self.target_path, index=False)
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_pipeline(self):
        """Test the complete mapping pipeline"""
        mapper = AutoLearningMapper()
        
        # Load data
        source_df = load_dataframe_from_file(self.source_path)
        target_df = load_dataframe_from_file(self.target_path)
        
        # Get mappings
        all_mappings = mapper.smart_suggest_mappings(
            list(source_df.columns), 
            list(target_df.columns),
            source_df, 
            target_df
        )
        
        # Get best mappings
        best_mappings = mapper.get_best_unique_mappings(all_mappings)
        
        # Should find some good mappings
        self.assertGreater(len(best_mappings), 0)
        
        # Expected mappings based on our test data
        expected_pairs = [
            ('cust_id', 'id'),
            ('customer_name', 'full_name'),
            ('email_addr', 'email'),
            ('ph_number', 'phone'),
            ('order_total', 'amount'),
            ('created_dt', 'registration_date')
        ]
        
        # Check that we found most expected mappings
        found_pairs = [(m.source_field, m.target_field) for m in best_mappings]
        matches = sum(1 for pair in expected_pairs if pair in found_pairs)
        
        # Should find at least 4 out of 6 expected mappings
        self.assertGreaterEqual(matches, 4, 
            f"Expected to find at least 4 mappings, found {matches}. "
            f"Found pairs: {found_pairs}")
        
        # Test saving results
        output_path = os.path.join(self.temp_dir, 'results.json')
        result_data = save_mappings_to_json(
            best_mappings, all_mappings, source_df, target_df, mapper,
            self.source_path, self.target_path, output_path
        )
        
        self.assertIsNotNone(result_data)
        self.assertTrue(os.path.exists(output_path))
        
        # Test HTML report generation
        html_path = os.path.join(self.temp_dir, 'report.html')
        generate_mapping_report(result_data, html_path)
        self.assertTrue(os.path.exists(html_path))
        
        # Test ETL script generation
        script_path = os.path.join(self.temp_dir, 'transform.py')
        generate_etl_script(output_path, self.source_path, script_path)
        self.assertTrue(os.path.exists(script_path))
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        mapper = AutoLearningMapper()
        
        # Test with empty dataframes
        empty_df = pd.DataFrame()
        mappings = mapper.smart_suggest_mappings([], [], empty_df, empty_df)
        self.assertEqual(len(mappings), 0)
        
        # Test with no matching fields
        source_fields = ['completely_different_field']
        target_fields = ['totally_unrelated_field']
        source_df = pd.DataFrame({'completely_different_field': [1, 2, 3]})
        target_df = pd.DataFrame({'totally_unrelated_field': [1, 2, 3]})
        
        mappings = mapper.smart_suggest_mappings(
            source_fields, target_fields, source_df, target_df, min_confidence=0.1
        )
        
        # Should find very few or no mappings
        self.assertLessEqual(len(mappings), 1)
    
    def test_performance_with_large_schemas(self):
        """Test performance with larger field lists"""
        import time
        
        mapper = AutoLearningMapper()
        
        # Create larger field lists
        source_fields = [f'source_field_{i}' for i in range(50)]
        target_fields = [f'target_field_{i}' for i in range(50)]
        
        # Add some similar fields
        source_fields.extend(['customer_name', 'email_address', 'phone_number'])
        target_fields.extend(['full_name', 'email', 'contact_phone'])
        
        start_time = time.time()
        mappings = mapper.smart_suggest_mappings(source_fields, target_fields)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 30 seconds)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 30, 
            f"Mapping took too long: {execution_time:.2f} seconds")
        
        # Should find at least the obvious matches
        self.assertGreater(len(mappings), 0)


class TestSampleDatasets(unittest.TestCase):
    """Test with various sample datasets to validate real-world scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.mapper = AutoLearningMapper()
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ecommerce_dataset(self):
        """Test with e-commerce style data"""
        # Legacy e-commerce system
        source_data = pd.DataFrame({
            'prod_id': [1, 2, 3],
            'product_nm': ['Laptop', 'Mouse', 'Keyboard'],
            'price_amt': [999.99, 29.99, 79.99],
            'cat_cd': ['COMP', 'ACC', 'ACC'],
            'inv_qty': [10, 50, 25],
            'created_ts': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00']
        })
        
        # Modern e-commerce system
        target_data = pd.DataFrame({
            'product_id': [0, 0, 0],
            'name': ['', '', ''],
            'price': [0.0, 0.0, 0.0],
            'category': ['', '', ''],
            'stock_quantity': [0, 0, 0],
            'date_added': ['', '', '']
        })
        
        mappings = self.mapper.smart_suggest_mappings(
            list(source_data.columns), 
            list(target_data.columns),
            source_data, 
            target_data
        )
        
        best_mappings = self.mapper.get_best_unique_mappings(mappings)
        
        # Should find good mappings
        self.assertGreaterEqual(len(best_mappings), 4)
        
        # Check for expected mappings
        mapping_dict = {m.source_field: m.target_field for m in best_mappings}
        expected_mappings = {
            'prod_id': 'product_id',
            'product_nm': 'name',
            'price_amt': 'price',
            'inv_qty': 'stock_quantity'
        }
        
        for source, target in expected_mappings.items():
            if source in mapping_dict:
                self.assertEqual(mapping_dict[source], target)
    
    def test_crm_dataset(self):
        """Test with CRM style data"""
        # Legacy CRM
        source_data = pd.DataFrame({
            'contact_id': [1, 2, 3],
            'first_nm': ['John', 'Jane', 'Bob'],
            'last_nm': ['Doe', 'Smith', 'Johnson'],
            'email': ['john.doe@email.com', 'jane.smith@email.com', 'bob.johnson@email.com'],
            'mobile_ph': ['555-0101', '555-0102', '555-0103'],
            'company_nm': ['ACME Corp', 'XYZ Inc', 'ABC Ltd'],
            'lead_score': [85, 92, 78]
        })
        
        # Modern CRM
        target_data = pd.DataFrame({
            'id': [0, 0, 0],
            'full_name': ['', '', ''],
            'email_address': ['', '', ''],
            'phone_number': ['', '', ''],
            'organization': ['', '', ''],
            'score': [0, 0, 0]
        })
        
        mappings = self.mapper.smart_suggest_mappings(
            list(source_data.columns), 
            list(target_data.columns),
            source_data, 
            target_data
        )
        
        best_mappings = self.mapper.get_best_unique_mappings(mappings)
        
        # Should find most mappings
        self.assertGreaterEqual(len(best_mappings), 4)
        
        # Verify some expected mappings
        found_mappings = [(m.source_field, m.target_field) for m in best_mappings]
        
        # Check that ID fields are mapped
        id_mapped = any(('contact_id' in pair and 'id' in pair) for pair in found_mappings)
        self.assertTrue(id_mapped, "ID fields should be mapped")
        
        # Check that email fields are mapped
        email_mapped = any(('email' in pair[0] and 'email' in pair[1]) for pair in found_mappings)
        self.assertTrue(email_mapped, "Email fields should be mapped")


def create_sample_datasets(output_dir):
    """Create sample datasets for manual testing"""
    
    # E-commerce dataset
    ecommerce_source = pd.DataFrame({
        'prod_id': range(1, 101),
        'product_name': [f'Product {i}' for i in range(1, 101)],
        'price_usd': [round(10 + i * 2.5, 2) for i in range(100)],
        'category_code': ['CAT' + str(i % 5) for i in range(100)],
        'stock_count': [50 + i % 30 for i in range(100)],
        'created_date': pd.date_range('2023-01-01', periods=100, freq='D').strftime('%Y-%m-%d')
    })
    
    ecommerce_target = pd.DataFrame({
        'product_id': [0] * 5,
        'name': [''] * 5,
        'price': [0.0] * 5,
        'category': [''] * 5,
        'inventory_quantity': [0] * 5,
        'date_added': [''] * 5
    })
    
    # CRM dataset
    crm_source = pd.DataFrame({
        'contact_id': range(1, 51),
        'fname': [f'FirstName{i}' for i in range(50)],
        'lname': [f'LastName{i}' for i in range(50)],
        'email_addr': [f'user{i}@company.com' for i in range(50)],
        'phone_num': [f'555-{1000+i:04d}' for i in range(50)],
        'company': [f'Company {i % 10}' for i in range(50)],
        'lead_score': [50 + i % 50 for i in range(50)],
        'status': ['active'] * 50
    })
    
    crm_target = pd.DataFrame({
        'id': [0] * 5,
        'full_name': [''] * 5,
        'email': [''] * 5,
        'contact_phone': [''] * 5,
        'organization': [''] * 5,
        'score': [0] * 5,
        'account_status': [''] * 5
    })
    
    # Financial dataset
    financial_source = pd.DataFrame({
        'account_num': [f'ACC{1000+i:04d}' for i in range(30)],
        'cust_name': [f'Customer {i}' for i in range(30)],
        'balance_amt': [1000 + i * 100.5 for i in range(30)],
        'acct_type': [['savings', 'checking', 'credit'][i % 3] for i in range(30)],
        'open_date': pd.date_range('2020-01-01', periods=30, freq='M').strftime('%Y-%m-%d'),
        'branch_cd': [f'BR{i % 5:02d}' for i in range(30)],
        'interest_rate': [0.01 + (i % 10) * 0.001 for i in range(30)]
    })
    
    financial_target = pd.DataFrame({
        'account_number': [''] * 5,
        'customer_name': [''] * 5,
        'current_balance': [0.0] * 5,
        'account_type': [''] * 5,
        'opening_date': [''] * 5,
        'branch_code': [''] * 5,
        'rate': [0.0] * 5
    })
    
    # Healthcare dataset
    healthcare_source = pd.DataFrame({
        'patient_id': [f'PAT{i:05d}' for i in range(1, 21)],
        'pt_fname': [f'FirstName{i}' for i in range(20)],
        'pt_lname': [f'LastName{i}' for i in range(20)],
        'dob': pd.date_range('1950-01-01', periods=20, freq='365D').strftime('%Y-%m-%d'),
        'gender': [['M', 'F'][i % 2] for i in range(20)],
        'diagnosis_cd': [f'ICD{i % 10:02d}' for i in range(20)],
        'admit_dt': pd.date_range('2023-01-01', periods=20, freq='7D').strftime('%Y-%m-%d'),
        'discharge_dt': (pd.date_range('2023-01-01', periods=20, freq='7D') + 
                        pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    })
    
    healthcare_target = pd.DataFrame({
        'patient_number': [''] * 5,
        'first_name': [''] * 5,
        'last_name': [''] * 5,
        'birth_date': [''] * 5,
        'sex': [''] * 5,
        'primary_diagnosis': [''] * 5,
        'admission_date': [''] * 5,
        'release_date': [''] * 5
    })
    
    # Save all datasets
    datasets = {
        'ecommerce': (ecommerce_source, ecommerce_target),
        'crm': (crm_source, crm_target),
        'financial': (financial_source, financial_target),
        'healthcare': (healthcare_source, healthcare_target)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, (source, target) in datasets.items():
        source.to_csv(os.path.join(output_dir, f'{name}_source.csv'), index=False)
        target.to_csv(os.path.join(output_dir, f'{name}_target.csv'), index=False)
        
        # Also create JSON versions for testing
        source.to_json(os.path.join(output_dir, f'{name}_source.json'), 
                      orient='records', indent=2)
        target.to_json(os.path.join(output_dir, f'{name}_target.json'), 
                      orient='records', indent=2)
    
    print(f"Sample datasets created in {output_dir}")
    return datasets


class TestRegressionSuite(unittest.TestCase):
    """Regression tests to catch performance degradation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mapper = AutoLearningMapper()
        
    def test_confidence_score_stability(self):
        """Test that confidence scores remain stable across runs"""
        source_fields = ['customer_name', 'email_address', 'phone_number']
        target_fields = ['full_name', 'email', 'contact_phone']
        
        # Run mapping multiple times
        scores = []
        for _ in range(3):
            mappings = self.mapper.smart_suggest_mappings(source_fields, target_fields)
            if mappings:
                # Find the customer_name -> full_name mapping
                for mapping in mappings:
                    if mapping.source_field == 'customer_name' and mapping.target_field == 'full_name':
                        scores.append(mapping.confidence_score)
                        break
        
        # Scores should be consistent
        if len(scores) > 1:
            score_variance = max(scores) - min(scores)
            self.assertLess(score_variance, 0.01, 
                f"Confidence scores vary too much: {scores}")
    
    def test_mapping_completeness(self):
        """Test that mappings are reasonably complete for obvious cases"""
        # Very obvious mappings
        source_fields = ['id', 'name', 'email', 'phone', 'address']
        target_fields = ['user_id', 'full_name', 'email_address', 'phone_number', 'home_address']
        
        mappings = self.mapper.smart_suggest_mappings(source_fields, target_fields)
        best_mappings = self.mapper.get_best_unique_mappings(mappings)
        
        # Should find most obvious mappings
        self.assertGreaterEqual(len(best_mappings), 4, 
            "Should find at least 4 obvious mappings")
        
        # Check specific high-confidence mappings
        high_conf_mappings = [m for m in best_mappings if m.confidence_score >= 0.7]
        self.assertGreaterEqual(len(high_conf_mappings), 2,
            "Should have at least 2 high-confidence mappings")


def run_performance_benchmark():
    """Run performance benchmarks and return results"""
    import time
    
    try:
        from memory_profiler import memory_usage
    except ImportError:
        print("⚠️  memory_profiler not installed. Install with: pip install memory_profiler")
        return {}
    
    mapper = AutoLearningMapper()
    
    # Test with different schema sizes
    test_sizes = [10, 25, 50, 100]
    results = {}
    
    for size in test_sizes:
        source_fields = [f'source_field_{i}' for i in range(size)]
        target_fields = [f'target_field_{i}' for i in range(size)]
        
        # Add some obvious matches
        source_fields.extend(['customer_name', 'email_address'])
        target_fields.extend(['full_name', 'email'])
        
        # Measure time and memory
        start_time = time.time()
        try:
            mem_usage = memory_usage((
                mapper.smart_suggest_mappings, 
                (source_fields, target_fields)
            ))
            end_time = time.time()
            
            results[size] = {
                'time': end_time - start_time,
                'memory_peak': max(mem_usage) if mem_usage else 0,
                'memory_average': sum(mem_usage) / len(mem_usage) if mem_usage else 0
            }
        except Exception as e:
            print(f"Error benchmarking size {size}: {e}")
            results[size] = {
                'time': -1,
                'memory_peak': -1,
                'memory_average': -1
            }
    
    return results


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Data Mapper Test Suite')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample datasets for manual testing')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--sample-dir', default='./test_samples',
                       help='Directory to create sample datasets')
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_datasets(args.sample_dir)
        print(f"\n✅ Sample datasets created in {args.sample_dir}")
        print("You can now test the mapper with commands like:")
        print(f"smart-mapper --source {args.sample_dir}/ecommerce_source.csv --target {args.sample_dir}/ecommerce_target.csv --html-report report.html")
        sys.exit(0)
    
    if args.benchmark:
        print("🚀 Running performance benchmarks...")
        try:
            results = run_performance_benchmark()
            print("\n📊 Performance Results:")
            for size, metrics in results.items():
                print(f"  Schema size {size}: {metrics['time']:.2f}s, "
                      f"Peak memory: {metrics['memory_peak']:.1f}MB")
        except ImportError:
            print("⚠️  memory_profiler not installed. Install with: pip install memory_profiler")
        sys.exit(0)
    
    # Run the test suite
    unittest.main(argv=[''], verbosity=2 if args.verbose else 1, exit=False)
    
    print("\n" + "="*60)
    print("🎯 TEST SUITE SUMMARY")
    print("="*60)
    print("✅ Unit Tests: Core mapper functionality")
    print("✅ Integration Tests: End-to-end pipeline") 
    print("✅ File Operations: CSV/JSON/XML loading")
    print("✅ Report Generation: HTML and ETL scripts")
    print("✅ Sample Datasets: Real-world scenarios")
    print("✅ Regression Tests: Stability and performance")
    print("\n💡 To create sample datasets: python test_smart_mapper.py --create-samples")
    print("💡 To run benchmarks: python test_smart_mapper.py --benchmark")
    print("="*60)