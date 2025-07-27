import json
import os
import re
from collections import defaultdict

def _get_css():
    """Returns the CSS for the HTML report."""
    return """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; padding-bottom: 10px; }
        h1, h2 { color: #569cd6; margin: 0; }
        .controls { display: flex; gap: 20px; align-items: center; }
        .filter-group { display: flex; gap: 10px; align-items: center; background-color: #252526; padding: 5px 10px; border-radius: 5px; }
        .export-btn { background-color: #0e639c; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; }
        .export-btn:hover { background-color: #1177bb; }
        .mapper-container { display: flex; position: relative; justify-content: space-between; padding: 20px; background-color: #252526; border-radius: 8px; margin-top: 20px; }
        .column { width: 45%; }
        ul { list-style-type: none; padding-left: 20px; }
        li { background-color: #333; border: 1px solid #444; padding: 8px 12px; margin-bottom: 5px; border-radius: 4px; transition: all 0.2s ease-in-out; cursor: default; }
        li.unmapped { border-left: 3px solid #d16d71; opacity: 0.6; }
        li.conflict { border-left: 3px solid #ce9178; }
        li.hidden-by-filter { opacity: 0.3; }
        #connections-svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        .tooltip { position: absolute; display: none; background-color: #1e1e1e; border: 1px solid #569cd6; padding: 10px; border-radius: 6px; pointer-events: none; z-index: 100; max-width: 300px; font-size: 12px; }
        .tooltip-reasoning { color: #ce9178; margin-top: 5px; }
    """

def _get_js(mappings, unmapped_source, unmapped_target, conflicting_targets):
    """Returns the JavaScript for the report, with data embedded."""
    mappings_json = json.dumps(mappings)
    unmapped_source_json = json.dumps(unmapped_source)
    unmapped_target_json = json.dumps(unmapped_target)
    conflicting_targets_json = json.dumps(conflicting_targets)

    return f"""
        // Load html2canvas library for exporting
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
        document.head.appendChild(script);

        window.onload = () => {{
            const mappings = {mappings_json};
            const unmappedSource = new Set({unmapped_source_json});
            const unmappedTarget = new Set({unmapped_target_json});
            const conflictingTargets = new Set({conflicting_targets_json});

            const svg = document.getElementById('connections-svg');
            const container = document.querySelector('.mapper-container');
            const tooltip = document.getElementById('tooltip');
            
            const filters = {{
                high: document.getElementById('filter-high'),
                medium: document.getElementById('filter-medium'),
                low: document.getElementById('filter-low')
            }};

            if (!container) return;

            const getVisibleMappings = () => {{
                return mappings.filter(m => {{
                    const score = m.confidence_score;
                    if (score >= 0.7 && filters.high.checked) return true;
                    if (score >= 0.5 && score < 0.7 && filters.medium.checked) return true;
                    if (score < 0.5 && filters.low.checked) return true;
                    return false;
                }});
            }};

            const drawLines = () => {{
                svg.innerHTML = ''; // Clear previous lines
                const containerRect = container.getBoundingClientRect();
                const visibleMappings = getVisibleMappings();
                const visibleSourceFields = new Set(visibleMappings.map(m => m.source_field));
                const visibleTargetFields = new Set(visibleMappings.map(m => m.target_field));

                // Toggle visibility class on list items
                document.querySelectorAll('#source-column li, #target-column li').forEach(li => {{
                    const fieldName = li.dataset.field;
                    if (unmappedSource.has(fieldName) || unmappedTarget.has(fieldName)) {{
                        li.classList.add('unmapped');
                    }} else if (conflictingTargets.has(fieldName)) {{
                        li.classList.add('conflict');
                    }}

                    if (!visibleSourceFields.has(fieldName) && !visibleTargetFields.has(fieldName) && !unmappedSource.has(fieldName) && !unmappedTarget.has(fieldName)) {{
                         li.classList.add('hidden-by-filter');
                    }} else {{
                         li.classList.remove('hidden-by-filter');
                    }}
                }});

                visibleMappings.forEach(mapping => {{
                    const sourceId = `source-${{mapping.source_field.replace(/[^a-zA-Z0-9]/g, '-')}}`;
                    const targetId = `target-${{mapping.target_field.replace(/[^a-zA-Z0-9]/g, '-')}}`;

                    const sourceEl = document.getElementById(sourceId);
                    const targetEl = document.getElementById(targetId);

                    if (sourceEl && targetEl) {{
                        const sourceRect = sourceEl.getBoundingClientRect();
                        const targetRect = targetEl.getBoundingClientRect();

                        const x1 = sourceRect.right - containerRect.left;
                        const y1 = sourceRect.top + sourceRect.height / 2 - containerRect.top;
                        const x2 = targetRect.left - containerRect.left;
                        const y2 = targetRect.top + targetRect.height / 2 - containerRect.top;

                        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        line.setAttribute('x1', x1);
                        line.setAttribute('y1', y1);
                        line.setAttribute('x2', x2);
                        line.setAttribute('y2', y2);

                        let color = '#d16d71'; // low
                        if (mapping.confidence_score >= 0.7) color = '#6a9955'; // high
                        else if (mapping.confidence_score >= 0.5) color = '#d7ba7d'; // medium

                        line.setAttribute('stroke', color);
                        line.setAttribute('stroke-width', '2');
                        line.style.transition = 'stroke 0.3s';
                        
                        const showTooltip = (e) => {{
                            tooltip.style.display = 'block';
                            tooltip.innerHTML = `
                                <strong>Confidence: ${{mapping.confidence_score.toFixed(3)}}</strong> (${{mapping.similarity_type}})<br>
                                Sample: <em>${{mapping.sample_source_value || 'N/A'}}</em>
                                <div class="tooltip-reasoning">${{mapping.reasoning}}</div>
                            `;
                            tooltip.style.left = `${{e.pageX + 15}}px`;
                            tooltip.style.top = `${{e.pageY + 15}}px`;
                        }};

                        const hideTooltip = () => {{ tooltip.style.display = 'none'; }};

                        sourceEl.addEventListener('mousemove', showTooltip);
                        sourceEl.addEventListener('mouseout', hideTooltip);
                        targetEl.addEventListener('mousemove', showTooltip);
                        targetEl.addEventListener('mouseout', hideTooltip);

                        svg.appendChild(line);
                    }}
                }});
            }};

            // Initial draw and event listeners
            drawLines();
            window.addEventListener('resize', drawLines);
            Object.values(filters).forEach(filter => filter.addEventListener('change', drawLines));
            
            // Export functionality
            document.getElementById('export-btn').addEventListener('click', () => {{
                if (typeof html2canvas === 'undefined') {{
                    alert('Export library is not loaded yet. Please try again in a moment.');
                    return;
                }}
                html2canvas(document.querySelector(".mapper-container")).then(canvas => {{
                    const link = document.createElement('a');
                    link.download = 'mapping-visualization.png';
                    link.href = canvas.toDataURL();
                    link.click();
                }});
            }});
        }};
    """

def _build_target_hierarchy(fields):
    """Builds a nested dictionary from dot-separated field paths."""
    hierarchy = {}
    for field in fields:
        parts = field.split('.')
        node = hierarchy
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]
    return hierarchy

def _sanitize_id(text: str) -> str:
    """Sanitizes a string to be used as a valid CSS ID component."""
    return re.sub(r'[^a-zA-Z0-9]', '-', text)

def _render_source_list(fields):
    """Renders the HTML list for source fields."""
    html = '<ul>'
    for field in sorted(fields):
        elem_id = f"source-{_sanitize_id(field)}"
        html += f'<li id="{elem_id}" data-field="{field}"><span>{field}</span></li>'
    html += '</ul>'
    return html

def _render_target_tree(hierarchy, path_prefix=''):
    """Recursively renders the HTML list for the target field hierarchy."""
    if not hierarchy:
        return ""
    html = '<ul>'
    for key, sub_hierarchy in sorted(hierarchy.items()):
        current_path = f"{path_prefix}{key}"
        elem_id = f"target-{_sanitize_id(current_path)}"

        html += f'<li id="{elem_id}" data-field="{current_path}">'
        html += f'<span>{key}</span>'
        if sub_hierarchy:
            html += _render_target_tree(sub_hierarchy, path_prefix=current_path + '.')
        html += '</li>'
    html += '</ul>'
    return html

def generate_mapping_report(data: dict, output_path: str):
    """Generates a self-contained HTML report to visualize mappings."""
    
    recommended_mappings = data.get('recommended_mappings', [])
    unmapped_fields_data = data.get('unmapped_fields', {})
    unmapped_source = unmapped_fields_data.get('source', [])
    unmapped_target = unmapped_fields_data.get('target', [])
    conflicting_mappings = data.get('conflicting_mappings', [])
    conflicting_targets = list(set(m['target_field'] for m in conflicting_mappings))

    # Get a complete list of all source and target fields for rendering the full lists
    all_source_fields = sorted(list(set(m['source_field'] for m in recommended_mappings) | set(unmapped_source)))
    all_target_fields = sorted(list(set(m['target_field'] for m in recommended_mappings) | set(unmapped_target) | set(conflicting_targets)))

    # Build the target hierarchy from ALL target field names (strings)
    target_hierarchy = _build_target_hierarchy(all_target_fields)

    html_css = _get_css()
    html_js = _get_js(recommended_mappings, unmapped_source, unmapped_target, conflicting_targets)
    source_html = _render_source_list(all_source_fields)
    target_html = _render_target_tree(target_hierarchy)
    
    source_file = data.get('metadata', {}).get('source_file', 'Source')
    target_file = data.get('metadata', {}).get('target_file', 'Target')

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Data Mapping Report</title>
        <style>{html_css}</style>
    </head>
    <body>
        <div class="header">
            <h1>Mapping Report: {source_file} → {target_file}</h1>
            <div class="controls">
                <div class="filter-group">
                    <span>Filter:</span>
                    <label><input type="checkbox" id="filter-high" checked> High</label>
                    <label><input type="checkbox" id="filter-medium" checked> Medium</label>
                    <label><input type="checkbox" id="filter-low" checked> Low</label>
                </div>
                <button id="export-btn" class="export-btn">Export as PNG</button>
            </div>
        </div>
        <div id="tooltip" class="tooltip"></div>
        <div class="mapper-container">
            <div class="column" id="source-column">
                <h2>Source Fields</h2>
                {source_html}
            </div>
            <svg id="connections-svg"></svg>
            <div class="column" id="target-column">
                <h2>Target Fields</h2>
                {target_html}
            </div>
        </div>
        <script>{html_js}</script>
    </body>
    </html>
    """
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    except IOError as e:
        print(f"❌ Error writing HTML report to '{output_path}': {e}")