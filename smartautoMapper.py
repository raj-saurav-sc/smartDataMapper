import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import warnings
import re
from typing import List, Dict, Tuple, Optional
import argparse
import csv
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import Levenshtein 

@dataclass
class FieldMapping:
    source_field: str
    target_field: str
    confidence_score: float
    similarity_type: str
    reasoning: str
    string_similarity: float
    semantic_similarity: float
    pattern_match: str
    data_type_match: float
    abbreviation_match: float
    semantic_group_match: float

class AutoLearningMapper:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
        
        # Auto-discovered patterns (learned from data)
        self.learned_patterns = defaultdict(list)
        self.abbreviation_dict = {}
        self.synonym_groups = defaultdict(set)
        
        # Common abbreviations that can be auto-detected
        self.common_abbreviations = {
            'addr': 'address', 'ph': 'phone', 'tel': 'telephone', 'mob': 'mobile',
            'num': 'number', 'nbr': 'number', 'qty': 'quantity', 'amt': 'amount',
            'desc': 'description', 'cd': 'code', 'nm': 'name', 'dt': 'date',
            'tm': 'time', 'sts': 'status', 'flg': 'flag', 'id': 'identifier',
            'ref': 'reference', 'bal': 'balance', 'tot': 'total', 'cnt': 'count',
            'val': 'value', 'max': 'maximum', 'min': 'minimum', 'avg': 'average'
        }

    def auto_discover_patterns(self, all_field_names: List[str], all_columns: Optional[Dict[str, pd.Series]] = None):
        """Automatically discover patterns from field names and data"""
        
        # 1. Auto-detect abbreviations by looking for similar roots
        self._discover_abbreviations(all_field_names)
        
        # 2. Group semantically similar fields using data if available
        self._discover_semantic_groups(all_field_names, all_columns=all_columns)
        
        # 3. Learn common suffixes/prefixes
        self._discover_naming_patterns(all_field_names)
        
        print(f"🤖 Auto-discovered {len(self.abbreviation_dict)} abbreviation patterns")
        print(f"🤖 Auto-discovered {len(self.synonym_groups)} semantic groups (data-enhanced)")

    def _create_data_signature(self, column: pd.Series) -> str:
        """Create a representative string signature from column data."""
        if column.empty or column.dropna().empty:
            return ""
        # Sample up to 50 non-null values for the signature
        sample = column.dropna().sample(n=min(50, len(column.dropna())))
        # Convert to string and join, ensuring variety but not excessive length
        return " ".join(sample.astype(str))[:1000] # Cap length
    
    def _discover_abbreviations(self, field_names: List[str]):
        """Auto-detect abbreviations by finding short words that match longer ones"""
        
        # Start with common abbreviations
        self.abbreviation_dict.update(self.common_abbreviations)
        
        # Clean field names for analysis
        clean_names = []
        for name in field_names:
            clean = re.sub(r'[_\-\.]', ' ', name.lower())
            words = clean.split()
            clean_names.extend(words)
        
        word_counts = Counter(clean_names)
        
        # Find potential abbreviations (short words that could expand to longer ones)
        for short_word in word_counts:
            if len(short_word) <= 4 and word_counts[short_word] > 1:
                # Look for longer words that start with or contain the short word
                candidates = []
                for long_word in word_counts:
                    if (len(long_word) > len(short_word) and 
                        (long_word.startswith(short_word) or short_word in long_word)):
                        candidates.append((long_word, self._calculate_abbreviation_score(short_word, long_word)))
                
                if candidates:
                    # Take the best candidate
                    best_match = max(candidates, key=lambda x: x[1])
                    if best_match[1] > 0.6:
                        self.abbreviation_dict[short_word] = best_match[0]
    
    def _calculate_abbreviation_score(self, short: str, long: str) -> float:
        """Calculate how likely short is an abbreviation of long"""
        if short == long:
            return 1.0
        
        # Check if short is a prefix
        if long.startswith(short):
            return 0.8
        
        # Check if short is consonants of long
        consonants = re.sub(r'[aeiou]', '', long)
        if short == consonants[:len(short)]:
            return 0.7
        
        # Check if short matches first letters of words in long
        if '_' in long or ' ' in long:
            words = re.split(r'[_\s]', long)
            initials = ''.join([w[0] for w in words if w])
            if short == initials:
                return 0.9
        
        return 0.0
    
    def _discover_semantic_groups(self, field_names: List[str], all_columns: Optional[Dict[str, pd.Series]] = None):
        """Group semantically similar fields using embeddings of names and data."""
        
        # 1. Get embeddings for all field names
        preprocessed_names = [self.preprocess_field_name(f) for f in field_names]
        name_embeddings = self.model.encode(preprocessed_names)
        similarity_matrix = cosine_similarity(name_embeddings) # Start with name similarity
        
        # 2. Enhance with data embeddings if data is available
        if all_columns:
            data_signatures = [self._create_data_signature(all_columns.get(field, pd.Series(dtype='object'))) for field in field_names]
            
            # Only proceed if we have meaningful signatures
            if any(sig for sig in data_signatures):
                data_embeddings = self.model.encode(data_signatures)
                data_similarity_matrix = cosine_similarity(data_embeddings)
                
                # Combine similarities, giving more weight to data similarity
                # as it's a stronger indicator of a functional match.
                similarity_matrix = 0.4 * similarity_matrix + 0.6 * data_similarity_matrix
        
        # Find groups of highly similar fields
        threshold = 0.75 # Increased threshold as data can create stronger signals
        visited = set()
        
        for i, field1 in enumerate(field_names):
            if i in visited:
                continue
                
            group = {field1}
            visited.add(i)
            
            for j, field2 in enumerate(field_names):
                if j != i and j not in visited and similarity_matrix[i][j] > threshold:
                    group.add(field2)
                    visited.add(j)
            
            if len(group) > 1:
                # Create a group key based on the most common root
                group_key = self._find_common_root(list(group))
                self.synonym_groups[group_key] = group
    
    def _find_common_root(self, words: List[str]) -> str:
        """Find the common root among a group of words"""
        if not words:
            return "unknown"
        
        # Find the longest common substring
        common = words[0].lower()
        for word in words[1:]:
            word_lower = word.lower()
            new_common = ""
            for i in range(min(len(common), len(word_lower))):
                if common[i] == word_lower[i]:
                    new_common += common[i]
                else:
                    break
            common = new_common
        
        return common if len(common) > 2 else min(words, key=len)
    
    def _discover_naming_patterns(self, field_names: List[str]):
        """Learn common naming patterns (prefixes, suffixes)"""
        
        prefixes = defaultdict(list)
        suffixes = defaultdict(list)
        
        for field in field_names:
            clean_field = re.sub(r'[_\-\.]', ' ', field.lower())
            words = clean_field.split()
            
            if len(words) > 1:
                # Learn prefix patterns
                prefix = words[0]
                rest = ' '.join(words[1:])
                prefixes[prefix].append(rest)
                
                # Learn suffix patterns
                suffix = words[-1]
                beginning = ' '.join(words[:-1])
                suffixes[suffix].append(beginning)
        
        # Store patterns that appear frequently
        self.learned_patterns['prefixes'] = {k: v for k, v in prefixes.items() if len(v) > 1}
        self.learned_patterns['suffixes'] = {k: v for k, v in suffixes.items() if len(v) > 1}
    
    def enhanced_string_similarity(self, str1: str, str2: str) -> Tuple[float, float]:
        """Enhanced string similarity with abbreviation and pattern awareness"""
        
        str1_clean = self.preprocess_field_name(str1)
        str2_clean = self.preprocess_field_name(str2)
        
        # Basic similarity
        basic_sim = self._basic_string_similarity(str1_clean, str2_clean)
        
        # Abbreviation matching
        abbrev_sim = self._abbreviation_similarity(str1_clean, str2_clean)
        
        return max(basic_sim, abbrev_sim), abbrev_sim
    
    def _abbreviation_similarity(self, str1: str, str2: str) -> float:
        """Check if one string is an abbreviation of another"""
        
        words1 = str1.split()
        words2 = str2.split()
        
        # Expand abbreviations
        expanded1 = [self.abbreviation_dict.get(word, word) for word in words1]
        expanded2 = [self.abbreviation_dict.get(word, word) for word in words2]
        
        # Use Jaccard similarity on the expanded word sets for a more nuanced score.
        # This avoids overly generous scores for partial matches.
        set1, set2 = set(expanded1), set(expanded2)
        
        if not set1 or not set2:
            return 0.0
        
        if set1 == set2:
            return 1.0 # Perfect match after expansion
        
        # Jaccard similarity is a good base
        overlap = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_sim = overlap / union if union > 0 else 0.0

        # Boost score for subset relationships, as they are strong indicators
        if set1.issubset(set2) or set2.issubset(set1):
            # The boost is proportional to how much "room" there is to improve
            return jaccard_sim + (1 - jaccard_sim) * 0.5
            
        return jaccard_sim
    
    def _basic_string_similarity(self, str1: str, str2: str) -> float:
        """Basic string similarity using multiple metrics"""
        
        if str1 == str2:
            return 1.0
        
        # Levenshtein distance
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        lev_sim = 1 - (Levenshtein.distance(str1, str2) / max_len)
        
        # Jaccard similarity
        words1, words2 = set(str1.split()), set(str2.split())
        if not words1 or not words2:
            return lev_sim
        
        jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Combine metrics
        return max(lev_sim, jaccard_sim)
    
    def preprocess_field_name(self, field_name: str) -> str:
        """Enhanced preprocessing with learned patterns"""
        clean_name = field_name.lower().strip()
        
        # Remove common prefixes/suffixes based on learned patterns
        for prefix in self.learned_patterns.get('prefixes', {}):
            if clean_name.startswith(prefix + '_'):
                clean_name = clean_name[len(prefix + '_'):]
                break
        
        # Replace separators with spaces
        clean_name = re.sub(r'[_\-\.]', ' ', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        return clean_name
    
    def smart_suggest_mappings(self, source_fields: List[str], target_fields: List[str], 
                              source_df: Optional[pd.DataFrame] = None, 
                              target_df: Optional[pd.DataFrame] = None,
                              min_confidence: float = 0.15) -> List[FieldMapping]:
        """Smart mapping with auto-learned patterns"""
        
        # First, learn patterns from all fields
        all_fields = source_fields + target_fields
        all_columns = {}
        if source_df is not None:
            all_columns.update(source_df.to_dict('series'))
        if target_df is not None:
            all_columns.update(target_df.to_dict('series'))
        if not all_columns:
            all_columns = None
        self.auto_discover_patterns(all_fields, all_columns=all_columns)
        
        all_mappings = []

        # Pre-calculate all semantic similarities for efficiency
        processed_source = [self.preprocess_field_name(f) for f in source_fields]
        processed_target = [self.preprocess_field_name(f) for f in target_fields]
        embeddings = self.model.encode(processed_source + processed_target)
        n_source = len(source_fields)
        semantic_similarities = cosine_similarity(embeddings[:n_source], embeddings[n_source:])
        
        # Calculate enhanced similarities
        for i, src_field in enumerate(source_fields):
            for j, tgt_field in enumerate(target_fields):
                
                # Enhanced string similarity
                string_sim, abbrev_sim = self.enhanced_string_similarity(src_field, tgt_field)
                
                # Semantic similarity (from pre-calculated matrix)
                semantic_sim = semantic_similarities[i][j]
                
                # Data type similarity (placeholder - would use actual data)
                data_type_sim = self._estimate_data_type_similarity(src_field, tgt_field, source_df, target_df)
                
                # Check for semantic group match (strong signal)
                semantic_group_sim = self._semantic_group_similarity(src_field, tgt_field)
                
                # Combined score with rebalanced weighting for semantic groups
                combined_score = (
                    0.25 * string_sim +          # String similarity
                    0.20 * float(semantic_sim) + # Base semantic similarity  
                    0.15 * abbrev_sim +          # Abbreviation bonus
                    0.10 * data_type_sim +       # Data type similarity
                    0.30 * semantic_group_sim    # Semantic group bonus
                )
                
                if combined_score >= min_confidence:
                    # Determine similarity type
                    if semantic_group_sim > 0.9:
                        sim_type = "Semantic Group Match"
                        reasoning = "Fields were grouped based on data and name similarity"
                    elif string_sim > 0.8:
                        sim_type = "Exact/Close Match"
                        reasoning = "Field names are very similar"
                    elif abbrev_sim > 0.7 or (abbrev_sim > 0.4 and string_sim > 0.6):
                        sim_type = "Abbreviation Match"
                        reasoning = "One field appears to be abbreviation of the other"
                    elif semantic_sim > 0.6 and combined_score > 0.4:
                        sim_type = "Semantic Match"
                        reasoning = "Fields have similar semantic meaning"
                    else:
                        sim_type = "Pattern Match"
                        reasoning = "Fields share common patterns"
                    
                    mapping = FieldMapping(
                        source_field=src_field,
                        target_field=tgt_field,
                        confidence_score=float(combined_score),
                        similarity_type=sim_type,
                        reasoning=reasoning,
                        string_similarity=float(string_sim),
                        semantic_similarity=float(semantic_sim),
                        pattern_match=f"auto-learned",
                        data_type_match=float(data_type_sim),
                        abbreviation_match=float(abbrev_sim),
                        semantic_group_match=float(semantic_group_sim)
                    )
                    all_mappings.append(mapping)
        
        # Sort by confidence
        all_mappings.sort(key=lambda x: x.confidence_score, reverse=True)
        return all_mappings
    
    def _infer_column_data_type(self, column: pd.Series) -> str:
        """Infer data type from the actual data in a pandas Series."""
        # Use a sample for performance on large datasets
        sample = column.dropna().sample(n=min(100, len(column.dropna())))
        if sample.empty:
            return 'unknown'

        # Attempt to convert to numeric, allowing for some non-numeric values and
        # cleaning strings that are functionally numeric (like phone numbers)
        sample_cleaned = sample.astype(str).str.replace(r'[^\d\.]', '', regex=True)
        numeric_sample = pd.to_numeric(sample_cleaned, errors='coerce')
        if numeric_sample.notna().sum() / len(sample) > 0.9:
            return 'numeric'

        # Attempt to convert to datetime, suppressing expected format inference warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datetime_sample = pd.to_datetime(sample, errors='coerce')
        if datetime_sample.notna().sum() / len(sample) > 0.9:
            return 'datetime'
        
        # Check for boolean
        if sample.isin([True, False, 0, 1]).all():
            return 'boolean'

        return 'string'

    def _get_type_from_name(self, field: str) -> str:
        """Fallback to estimate type from field name."""
        numeric_indicators = ['amount', 'total', 'count', 'number', 'qty', 'balance', 'sum']
        date_indicators = ['date', 'time', 'created', 'updated', 'modified']
        text_indicators = ['name', 'description', 'comment', 'note', 'title']
        field_lower = field.lower()
        if any(indicator in field_lower for indicator in numeric_indicators):
            return 'numeric'
        elif any(indicator in field_lower for indicator in date_indicators):
            return 'datetime'
        elif any(indicator in field_lower for indicator in text_indicators):
            return 'string'
        return 'unknown'

    def _estimate_data_type_similarity(self, field1: str, field2: str, 
                                       source_df: Optional[pd.DataFrame] = None, 
                                       target_df: Optional[pd.DataFrame] = None) -> float:
        """Estimate data type similarity, using actual data if available."""
        
        # Priority 1: Use actual data from DataFrames
        if source_df is not None and target_df is not None and field1 in source_df and field2 in target_df:
            type1 = self._infer_column_data_type(source_df[field1])
            type2 = self._infer_column_data_type(target_df[field2])
        # Priority 2: Fallback to name-based heuristics
        else:
            type1 = self._get_type_from_name(field1)
            type2 = self._get_type_from_name(field2)
        
        if type1 == type2 and type1 != 'unknown':
            return 1.0
        elif type1 == 'unknown' or type2 == 'unknown':
            return 0.5
        else:
            return 0.0

    def _semantic_group_similarity(self, field1: str, field2: str) -> float:
        """Check if two fields belong to the same auto-discovered semantic group."""
        for group in self.synonym_groups.values():
            if field1 in group and field2 in group:
                return 1.0  # They are in the same semantic group
        return 0.0

    def get_best_unique_mappings(self, all_mappings: List[FieldMapping]) -> List[FieldMapping]:
        """Get the best unique mappings (one-to-one) from all possible mappings"""
        used_sources = set()
        used_targets = set()
        best_mappings = []
        
        # Mappings are already sorted by confidence score
        for mapping in all_mappings:
            if (mapping.source_field not in used_sources and 
                mapping.target_field not in used_targets):
                best_mappings.append(mapping)
                used_sources.add(mapping.source_field)
                used_targets.add(mapping.target_field)
        
        return best_mappings

def save_mappings_to_json(
    best_mappings: List[FieldMapping], 
    all_mappings: List[FieldMapping],
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mapper: 'AutoLearningMapper',
    source_file_path: str,
    target_file_path: str,
    output_path: str
):
    """Saves a comprehensive mapping report to a JSON file and returns the data."""
    from datetime import datetime

    # 1. Enhance recommended mappings with data types and sample values
    enhanced_recommended = []
    for m in best_mappings:
        mapping_dict = asdict(m)

        # Add inferred data types for source and target
        if m.source_field in source_df:
            mapping_dict['source_data_type'] = mapper._infer_column_data_type(source_df[m.source_field])
        if m.target_field in target_df:
            mapping_dict['target_data_type'] = mapper._infer_column_data_type(target_df[m.target_field])
        
        # Add a sample value from the source for context
        sample_val = source_df[m.source_field].dropna().iloc[0] if not source_df[m.source_field].dropna().empty else None
        mapping_dict['sample_source_value'] = str(sample_val) if sample_val is not None else None
        
        enhanced_recommended.append(mapping_dict)

    # 2. Identify unmapped fields
    mapped_sources = {m.source_field for m in best_mappings}
    mapped_targets = {m.target_field for m in best_mappings}
    unmapped_source = [f for f in source_df.columns if f not in mapped_sources]
    unmapped_target = [f for f in target_df.columns if f not in mapped_targets]

    # 3. Identify conflicting/alternative mappings for manual review
    conflicting_mappings = []
    for m in all_mappings:
        is_in_best = any(bm.source_field == m.source_field and bm.target_field == m.target_field for bm in best_mappings)
        if not is_in_best and (m.source_field in mapped_sources or m.target_field in mapped_targets) and m.confidence_score > 0.3:
            conflicting_mappings.append(asdict(m))
    
    output_data = {
        "metadata": {
            "source_file": os.path.basename(source_file_path),
            "target_file": os.path.basename(target_file_path),
            "generated_at": datetime.now().isoformat(),
            "mapping_count": len(best_mappings),
            "unmapped_source_count": len(unmapped_source),
            "unmapped_target_count": len(unmapped_target),
            "conflicting_mappings_count": len(conflicting_mappings)
        },
        "recommended_mappings": enhanced_recommended,
        "unmapped_fields": {"source": unmapped_source, "target": unmapped_target},
        "conflicting_mappings": conflicting_mappings
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        return output_data
    except IOError as e:
        print(f"❌ Error saving mappings to '{output_path}': {e}")
        return None

# Import the beautiful display class
from visualize_mappings import VisualMappingDisplay
from report_generator import generate_mapping_report

def load_dataframe_from_file(file_path: str) -> pd.DataFrame:
    """Loads a DataFrame from a CSV, JSON, or XML file, detecting the format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    
    _, file_extension = os.path.splitext(file_path)
    
    try:
        if file_extension.lower() == '.csv':
            # Sniff to find the delimiter for robustness
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                try:
                    dialect = csv.Sniffer().sniff(f.read(2048))
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ','  # Default to comma if sniffing fails
            return pd.read_csv(file_path, delimiter=delimiter)
        elif file_extension.lower() == '.json':
            # Use json_normalize to handle nested JSON structures correctly.
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.json_normalize(data)
        elif file_extension.lower() == '.xml':
            try:
                return pd.read_xml(file_path)
            except ImportError:
                raise ImportError("Reading XML files requires the 'lxml' library. Please install it using 'pip install lxml'.")
        else:
            print(f"⚠️ Warning: Unknown file type '{file_extension}'. Attempting to read as CSV.")
            return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading or parsing file '{file_path}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="""
        Smart Data Mapper CLI.
        Automatically suggests mappings between two tabular data files (CSV, JSON, or XML).
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--source', required=True, help="Path to the source data file (e.g., source.csv)")
    parser.add_argument('--target', required=True, help="Path to the target data file (e.g., target.json)")
    parser.add_argument('--min-confidence', type=float, default=0.2, help="Minimum confidence score for mappings (default: 0.2).")
    parser.add_argument('--html-report', help="Path to save a visual HTML report of the mappings (e.g., report.html)")
    parser.add_argument('--output', help="Path to save the final recommended mappings (e.g., mappings.json)")
    args = parser.parse_args()

    try:
        print(f"🔄 Loading source file: {args.source}")
        source_df = load_dataframe_from_file(args.source)
        
        print(f"🔄 Loading target file: {args.target}")
        target_df = load_dataframe_from_file(args.target)
    except (FileNotFoundError, IOError) as e:
        print(f"❌ {e}")
        return

    mapper = AutoLearningMapper()
    display = VisualMappingDisplay()
    
    source_schema = list(source_df.columns)
    target_schema = list(target_df.columns)
    
    print("\n🤖 Running Auto-Learning Smart Data Mapper...")
    print(f"Source Schema: {source_schema}")
    print(f"Target Schema: {target_schema}")
    
    # Get ALL possible mappings using the schemas AND the dataframes
    all_mappings = mapper.smart_suggest_mappings(
        source_fields=source_schema, 
        target_fields=target_schema,
        source_df=source_df,
        target_df=target_df,
        min_confidence=args.min_confidence
    )

    # Get the best one-to-one mappings from the full list
    best_mappings = mapper.get_best_unique_mappings(all_mappings)
    
    # Use the beautiful display to show the recommended mappings
    print() # Add a newline for better spacing
    display.display_beautiful_mappings(
        best_mappings, 
        source_schema, 
        target_schema, 
        show_details=True,
        title="✅ RECOMMENDED MAPPINGS (BEST UNIQUE MATCHES)"
    )
    
    # Show discovered patterns
    display.print_section_header("AUTO-DISCOVERED PATTERNS", "🤖")
    print(f"Abbreviations: {dict(mapper.abbreviation_dict)}")
    print(f"Semantic groups: {len(mapper.synonym_groups)} groups found")

    # Save the mappings to a file if requested
    if args.output:
        output_data = save_mappings_to_json(
            best_mappings=best_mappings,
            all_mappings=all_mappings,
            source_df=source_df,
            target_df=target_df,
            mapper=mapper,
            source_file_path=args.source, # Corrected argument name
            target_file_path=args.target, # Corrected argument name
            output_path=args.output
        )
        if output_data:
            print(f"\n✅ Comprehensive mapping report saved to {args.output}")
            # Generate HTML report if requested
            if args.html_report:
                print(f"🎨 Generating HTML report...")
                generate_mapping_report(output_data, args.html_report)
                print(f"✅ Visual report saved to {args.html_report}")

if __name__ == "__main__":
    main()