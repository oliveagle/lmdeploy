#!/usr/bin/env python3
"""
Generate architecture diagrams showing the call chain and overhead for each mode.
"""

import graphviz

def create_mode_diagram(mode_name: str, layers: list, overheads: list) -> graphviz.Digraph:
    """Create a diagram for a specific mode."""
    dot = graphviz.Digraph(comment=f'{mode_name} Architecture')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    # Add layers
    for i, (layer, color) in enumerate(layers):
        dot.node(f'layer{i}', layer, fillcolor=color, fontsize='12')

    # Connect layers
    for i in range(len(layers) - 1):
        dot.edge(f'layer{i}', f'layer{i+1}', label='FFI/API Call')

    # Add overhead annotations
    with dot.subgraph(name='cluster_overhead') as c:
        c.attr(label='Overhead Sources', style='dashed')
        for i, overhead in enumerate(overheads):
            c.node(f'overhead{i}', overhead, shape='note', fillcolor='lightyellow')

    return dot


def generate_all_diagrams():
    """Generate diagrams for all three modes."""

    # Mode 1: Python Direct
    python_layers = [
        ('User Script (Python)', 'lightblue'),
        ('lmdeploy.turbomind (Python)', 'lightblue'),
        ('_turbomind.so (pybind11)', 'lightcoral'),
        ('libturbomind_c.so (C++)', 'lightcoral'),
        ('CUDA Kernels', 'lightgray'),
    ]

    python_overheads = [
        'Python object creation',
        'pybind11 list conversion',
        'TurboMind instance per call',
        'No connection pooling',
    ]

    python_dot = create_mode_diagram('Python Direct', python_layers, python_overheads)
    python_dot.render('python_direct_architecture', format='png', cleanup=True)
    print("Generated: python_direct_architecture.png")

    # Mode 2: Rust + C++
    rust_layers = [
        ('gRPC Client', 'lightgreen'),
        ('Rust Server (lmdeploy-rust)', 'lightgreen'),
        ('FFI (turbomind_c.rs)', 'lightcoral'),
        ('libturbomind_c.so (C++)', 'lightcoral'),
        ('CUDA Kernels', 'lightgray'),
    ]

    rust_overheads = [
        'gRPC serialization (per token!)',
        'TCP socket latency',
        'Protobuf encode/decode',
        'Memory allocation per token',
    ]

    rust_dot = create_mode_diagram('Rust + C++', rust_layers, rust_overheads)
    rust_dot.render('rust_purecpp_architecture', format='png', cleanup=True)
    print("Generated: rust_purecpp_architecture.png")

    # Mode 3: PyBridge
    pybridge_layers = [
        ('User Script (Python)', 'lightblue'),
        ('stdin/stdout (JSON IPC)', 'orange'),
        ('python_bridge.py (Python)', 'lightblue'),
        ('lmdeploy.turbomind (Python)', 'lightblue'),
        ('_turbomind.so (pybind11)', 'lightcoral'),
        ('libturbomind_c.so (C++)', 'lightcoral'),
        ('CUDA Kernels', 'lightgray'),
    ]

    pybridge_overheads = [
        'subprocess fork+exec',
        'JSON serialization (40KB+)',
        'Two Python interpreters',
        'Pipe buffering latency',
        'No subprocess reuse',
    ]

    pybridge_dot = create_mode_diagram('PyBridge', pybridge_layers, pybridge_overheads)
    pybridge_dot.render('pybridge_architecture', format='png', cleanup=True)
    print("Generated: pybridge_architecture.png")


def create_comparison_table():
    """Create a comparison diagram."""
    dot = graphviz.Digraph(comment='Mode Comparison')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='plaintext')

    # Comparison table
    table = """<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD BGCOLOR="lightgray">Metric</TD>
        <TD BGCOLOR="lightblue">Python Direct</TD>
        <TD BGCOLOR="lightgreen">Rust + C++</TD>
        <TD BGCOLOR="orange">PyBridge</TD>
    </TR>
    <TR>
        <TD>Call Depth</TD>
        <TD>4 layers</TD>
        <TD>3 layers</TD>
        <TD>6 layers</TD>
    </TR>
    <TR>
        <TD>Prefill Speed</TD>
        <TD>Medium</TD>
        <TD>High</TD>
        <TD>Low</TD>
    </TR>
    <TR>
        <TD>Decode Speed</TD>
        <TD>Medium</TD>
        <TD>Low-Medium</TD>
        <TD>Low</TD>
    </TR>
    <TR>
        <TD>Main Bottleneck</TD>
        <TD>pybind11 conversion</TD>
        <TD>gRPC per-token</TD>
        <TD>JSON + subprocess</TD>
    </TR>
    <TR>
        <TD>TTFT</TD>
        <TD>High (model load)</TD>
        <TD>Medium</TD>
        <TD>High (subprocess)</TD>
    </TR>
    <TR>
        <TD>Batching Support</TD>
        <TD>Yes (internal)</TD>
        <TD>No (per-token)</TD>
        <TD>Yes (internal)</TD>
    </TR>
</TABLE>>"""

    dot.node('comparison', table)
    dot.render('mode_comparison_table', format='png', cleanup=True)
    print("Generated: mode_comparison_table.png")


if __name__ == '__main__':
    print("Generating architecture diagrams...")
    print("Note: Requires graphviz to be installed (pip install graphviz)")
    try:
        generate_all_diagrams()
        create_comparison_table()
        print("\nAll diagrams generated successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure graphviz is installed:")
        print("  pip install graphviz")
        print("  sudo apt-get install graphviz  # For system library")
