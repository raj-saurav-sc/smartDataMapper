# Smart Data Mapper

A Python-based tool that uses machine learning and heuristics to automatically suggest mappings between different data schemas. It's designed to accelerate data integration, migration, and ETL tasks by providing intelligent, data-aware mapping recommendations.

The project evolves from simple rule-based matching to a sophisticated `AutoLearningMapper` that requires minimal configuration.

## ✨ Key Features

-   **Auto-Learning Engine**: Automatically discovers patterns, abbreviations, and semantic relationships from your data fields and content. No manual rule-writing needed.
-   **Multi-Format Support**: Natively reads and parses data from CSV, nested JSON, and XML files.
-   **Intelligent Matching**: Combines multiple techniques for high accuracy:
    -   Levenshtein distance and Jaccard similarity for string matching.
    -   Sentence-Transformers for deep semantic understanding of field names and content.
    -   Data-driven type inference (numeric, date, string, etc.).
-   **Interactive HTML Report**: Generates a self-contained, interactive visual report to:
    -   Visualize mappings with a tree-like structure.
    -   Hover over fields for detailed tooltips (confidence, reasoning, sample data).
    -   Filter mappings by confidence level.
    -   Highlight unmapped fields and potential conflicts.
    -   Export the view as a PNG image.
-   **Comprehensive JSON Output**: Exports a detailed JSON report with recommended mappings, unmapped fields, and potential conflicts for programmatic use.
-   **ETL Script Generation**: Automatically generates a standalone Python script to perform the data transformation based on the recommended mappings.
-   **Flexible Command-Line Interface**: Easy to integrate into automated workflows and data pipelines.

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+
-   `pip` for installing packages

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd smartDataMapper
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the package:**
    The project is packaged with `setup.py`. To install it along with all its dependencies, run:
    ```bash
    pip install .
    ```
    For developers who want to modify the code, install it in editable mode:
    ```bash
    pip install -e .
    ```

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

-   `--source`: (Required) Path to the source data file (CSV, JSON, or XML).
-   `--target`: (Required) Path to the target data file (CSV, JSON, or XML).
-   `--output`: (Optional) Path to save the comprehensive mapping results in JSON format.
-   `--html-report`: (Optional) Path to save the interactive visual report in HTML format.
-   `--min-confidence`: (Optional) The minimum confidence score for a mapping to be considered. Defaults to `0.2`.

## Project Structure

-   `smartautoMapper.py`: The main CLI application and the core `AutoLearningMapper` class logic.
-   `report_generator.py`: Generates the interactive HTML visualization from the mapping data.
-   `visualize_mappings.py`: A helper class for generating formatted, color-coded terminal output.
-   `enhanced_mapper.py` / `main.py`: Earlier versions of the mapper, kept for reference.