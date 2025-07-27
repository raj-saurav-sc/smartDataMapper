# Smart Data Mapper

A Python-based tool that uses machine learning and heuristics to automatically suggest mappings between different data schemas. It's designed to accelerate data integration, migration, and ETL tasks by providing intelligent, data-aware mapping recommendations.

The project evolves from simple rule-based matching to a sophisticated `AutoLearningMapper` that requires minimal configuration.

## ✨ Key Features

* **Auto-Learning Engine**: Automatically discovers patterns, abbreviations, and semantic relationships from your data fields and content. No manual rule-writing needed.
* **Multi-Format Support**: Natively reads and parses data from CSV, nested JSON, and XML files.
* **Intelligent Matching**: Combines multiple techniques for high accuracy:
   * Levenshtein distance and Jaccard similarity for string matching.
   * Sentence-Transformers for deep semantic understanding of field names and content.
   * Data-driven type inference (numeric, date, string, etc.).
* **Interactive HTML Report**: Generates a self-contained, interactive visual report to:
   * Visualize mappings with a tree-like structure.
   * Hover over fields for detailed tooltips (confidence, reasoning, sample data).
   * Filter mappings by confidence level.
   * Highlight unmapped fields and potential conflicts.
   * Export the view as a PNG image.
* **Comprehensive JSON Output**: Exports a detailed JSON report with recommended mappings, unmapped fields, and potential conflicts for programmatic use.
* **ETL Script Generation**: Automatically generates a standalone Python script to perform the data transformation based on the recommended mappings.
* **Flexible Command-Line Interface**: Easy to integrate into automated workflows and data pipelines.

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* `pip` for installing packages

### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd smartDataMapper
```

2. **Create a virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install the package:** The project is packaged with `setup.py`. To install it along with all its dependencies, run:

```bash
pip install .
```

For developers who want to modify the code, install it in editable mode:

```bash
pip install -e .
```

*Note:* `lxml` is required for XML file support and will be automatically installed.

## Usage

Run the mapper from the command line, providing the source and target data files. The tool will analyze the files and generate the mapping reports.

### Basic Example

```bash
smart-mapper \
  --source source.csv \
  --target target.json \
  --output recommended_mappings.json \
  --html-report mapping_report.html \
  --generate-script transform.py
```

### Command-Line Arguments

* `--source`: (Required) Path to the source data file (CSV, JSON, or XML).
* `--target`: (Required) Path to the target data file (CSV, JSON, or XML).
* `--output`: (Optional) Path to save the comprehensive mapping results in JSON format.
* `--html-report`: (Optional) Path to save the interactive visual report in HTML format.
* `--generate-script`: (Optional) Path to save a generated Python ETL script (e.g., transform.py). Requires --output to be set.
* `--min-confidence`: (Optional) The minimum confidence score for a mapping to be considered. Defaults to 0.2.

### Example 2: Handling Nested JSON Targets

The mapper automatically handles nested structures. If your target JSON has fields like customer.details.name, the tool will correctly identify and map them. The HTML report will visualize this hierarchy.

```bash
# target_schema.json might contain {"customer": {"details": {"name": null, "email": null}}}
smart-mapper \
  --source data/legacy_users.csv \
  --target data/target_schema.json \
  --html-report reports/nested_report.html
```

### Example 3: Using the Generated ETL Script

After running the mapper with --generate-script, you get a ready-to-use Python script.

1. **Generate the script:**

```bash
smart-mapper --source data/users.csv --target data/customers.json --generate-script transform.py --output reports/mappings.json
```

2. **Inspect transform.py:** You can open the generated file to see the logic. It's a standard Python script using Pandas.

3. **Run the transformation:**

```bash
python transform.py
```

4. **Check the output:** A new file, transformed_data.json, will be created in the same directory, containing the source data structured according to the target schema.

## 📊 Understanding the Output

The tool provides insights in four ways:

### 1. Terminal Output

Get an immediate, color-coded summary of the results, including recommended mappings, unmapped fields, and coverage statistics.

### 2. Interactive HTML Report (--html-report)

A powerful, self-contained HTML file for exploring the mappings visually.

- **Visual Connections**: Lines connect mapped fields, colored by confidence (Green: High, Yellow: Medium, Red: Low).
- **Rich Tooltips**: Hover over any field or line to see detailed metrics, the reasoning for the match, and a sample source value.
- **Filtering**: Use checkboxes to show or hide mappings based on their confidence level.
- **Conflict & Unmapped Highlighting**: Unmapped fields and fields with alternative mappings are clearly marked.

### 3. Comprehensive JSON Report (--output)

A machine-readable file perfect for programmatic use.

```json
{
  "metadata": {
    "source_file": "users.csv",
    "target_file": "customers.json",
    "mapping_count": 8,
    "unmapped_source_count": 0
  },
  "recommended_mappings": [
    {
      "source_field": "cust_name",
      "target_field": "full_name",
      "confidence_score": 0.95,
      "similarity_type": "Semantic Group Match",
      "reasoning": "Fields were grouped based on data and name similarity",
      "sample_source_value": "John Doe"
    }
  ],
  "unmapped_fields": { 
    "source": [], 
    "target": ["notes"] 
  },
  "conflicting_mappings": [
    {
      "source_field": "created_date",
      "target_field": "last_updated",
      "confidence_score": 0.65
    }
  ]
}
```

### 4. Generated ETL Script (--generate-script)

A standalone Python script that uses Pandas to perform the transformation defined in the mappings.json file. It serves as an excellent starting point for a production data pipeline.

## Project Structure

- `smartautoMapper.py`: The main CLI application and the core `AutoLearningMapper` class logic.
- `report_generator.py`: Generates the interactive HTML visualization from the mapping data.
- `visualize_mappings.py`: A helper class for generating formatted, color-coded terminal output.
- `etl_generator.py`: Generates the standalone Python ETL script.
- `enhanced_mapper.py` / `main.py`: Earlier versions of the mapper, kept for reference.
- `setup.py`: The package definition for easy installation via pip.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.