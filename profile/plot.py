import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: str) -> dict:
    """Load profiling results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_latency_comparison(all_results: list, output_path: str, labels: list = None) -> None:
    """Plot latency and GFLOPS comparison between eval and deploy modes."""
    # If all_results is a single list (old format), wrap it
    if not isinstance(all_results[0], list):
        all_results = [all_results]
        labels = labels or ["Device"]
    
    # Different colors for CPU vs CUDA, same line styles for native vs packed
    # Native PyTorch: solid line, Packed BitLinear: dashed line (more prominent)
    device_colors = {
        'CPU': '#3498db',    # Blue
        'CUDA': '#e74c3c',   # Red
    }
    
    markers = ['o', 's', 'D', '^']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot data for each device
    for idx, (results, label) in enumerate(zip(all_results, labels)):
        marker = markers[idx % len(markers)]
        
        # Get color for this device (default to first color if not found)
        device_color = device_colors.get(label, '#3498db')
        
        # Separate eval and deploy results
        eval_results = [r for r in results if r["mode"] == "eval"]
        deploy_results = [r for r in results if r["mode"] == "deploy"]
        
        # Extract batch sizes and latencies
        batch_sizes = [r["batch_size"] for r in eval_results]
        eval_latencies = [r["latency_ms"] for r in eval_results]
        deploy_latencies = [r["latency_ms"] for r in deploy_results]

        # Extract GFLOPS (throughput in GFLOP/s)
        eval_gflops = [r["gflops"] for r in eval_results]
        deploy_gflops = [r["gflops"] for r in deploy_results]
        
        if label == "CPU": 
            label_str = "CPU (x86)"
        elif label == "CUDA": 
            label_str = "CUDA (5090)"
        else: 
            label_str = label
        # Plot 1: Latency vs Batch Size
        ax1.plot(
            batch_sizes, eval_latencies,
            marker=marker, linewidth=2, markersize=6,
            label=f'{label_str} - Native Pytorch',
            color=device_color, alpha=0.8,
            linestyle='-'
        )
        ax1.plot(
            batch_sizes, deploy_latencies,
            marker=marker, linewidth=2, markersize=6,
            label=f'{label_str} - Packed BitLinear',
            color=device_color, alpha=0.8,
            linestyle='--', dashes=(5, 2)
        )
        
        # Plot 2: GFLOP/s vs Batch Size
        ax2.plot(
            batch_sizes, eval_gflops,
            marker=marker, linewidth=2, markersize=6,
            label=f'{label_str} - Native Pytorch',
            color=device_color, alpha=0.8,
            linestyle='-'
        )
        ax2.plot(
            batch_sizes, deploy_gflops,
            marker=marker, linewidth=2, markersize=6,
            label=f'{label_str} - Packed BitLinear',
            color=device_color, alpha=0.8,
            linestyle='--', dashes=(5, 2)
        )
    
    # Configure Plot 1 (Latency)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Latency vs Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    # Set explicit ticks to show actual batch size values
    batch_size_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ax1.set_xticks(batch_size_ticks)
    ax1.set_xticklabels([str(bs) for bs in batch_size_ticks])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    
    # Configure Plot 2 (GFLOPS)
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (GFLOP/s)', fontsize=12)
    ax2.set_title('Compute Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    # Set explicit ticks to show actual batch size values
    ax2.set_xticks(batch_size_ticks)
    ax2.set_xticklabels([str(bs) for bs in batch_size_ticks])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Latency + GFLOPS plot saved to {output_path}")
    plt.close()


def plot_throughput_comparison(all_results: list, output_path: str, labels: list = None) -> None:
    """Plot throughput comparison between eval and deploy modes, optionally across multiple devices."""
    # If all_results is a single list (old format), wrap it
    if not isinstance(all_results[0], list):
        all_results = [all_results]
        labels = labels or ["Device"]
    
    # Different colors for CPU vs CUDA, same line styles for native vs packed
    # Native PyTorch: solid line, Packed BitLinear: dashed line (more prominent)
    device_colors = {
        'CPU': '#3498db',    # Blue
        'CUDA': '#e74c3c',   # Red
    }
    
    markers = ['o', 's', 'D', '^']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot data for each device
    for idx, (results, label) in enumerate(zip(all_results, labels)):
        marker = markers[idx % len(markers)]
        
        # Get color for this device (default to first color if not found)
        device_color = device_colors.get(label, '#3498db')
        
        # Separate eval and deploy results
        eval_results = [r for r in results if r["mode"] == "eval"]
        deploy_results = [r for r in results if r["mode"] == "deploy"]
        
        # Extract batch sizes and metrics
        batch_sizes = [r["batch_size"] for r in eval_results]
        eval_throughput = [r["throughput_samples_per_sec"] for r in eval_results]
        deploy_throughput = [r["throughput_samples_per_sec"] for r in deploy_results]
        eval_gflops = [r["gflops"] for r in eval_results]
        deploy_gflops = [r["gflops"] for r in deploy_results]
        
        # Plot 1: Throughput (samples/sec)
        ax1.plot(batch_sizes, eval_throughput, marker=marker, linewidth=2, 
                 markersize=6, label=f'{label} - Native Pytorch', color=device_color, alpha=0.8,
                 linestyle='-')
        ax1.plot(batch_sizes, deploy_throughput, marker=marker, linewidth=2, 
                 markersize=6, label=f'{label} - Packed BitLinear', color=device_color, alpha=0.8,
                 linestyle='--', dashes=(5, 2))
        
        # Plot 2: GFLOP/s
        ax2.plot(batch_sizes, eval_gflops, marker=marker, linewidth=2, 
                 markersize=6, label=f'{label} - Native Pytorch', color=device_color, alpha=0.8,
                 linestyle='-')
        ax2.plot(batch_sizes, deploy_gflops, marker=marker, linewidth=2, 
                 markersize=6, label=f'{label} - Packed BitLinear', color=device_color, alpha=0.8,
                 linestyle='--', dashes=(5, 2))
    
    # Configure Plot 1
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax1.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    # Set explicit ticks to show actual batch size values
    batch_size_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ax1.set_xticks(batch_size_ticks)
    ax1.set_xticklabels([str(bs) for bs in batch_size_ticks])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    
    # Configure Plot 2
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('GFLOP/s', fontsize=12)
    ax2.set_title('Compute Performance', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    # Set explicit ticks to show actual batch size values
    ax2.set_xticks(batch_size_ticks)
    ax2.set_xticklabels([str(bs) for bs in batch_size_ticks])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Throughput plot saved to {output_path}")
    plt.close()


def plot_memory_comparison(results: list, output_path: str) -> None:
    """Plot memory consumption comparison between eval and deploy modes."""
    # Separate eval and deploy results
    eval_results = [r for r in results if r["mode"] == "eval"]
    deploy_results = [r for r in results if r["mode"] == "deploy"]
    
    # Extract batch sizes and static model memory
    batch_sizes = [r["batch_size"] for r in eval_results]
    eval_model_memory = [r["model_memory_mb"] for r in eval_results]
    deploy_model_memory = [r["model_memory_mb"] for r in deploy_results]
    
    # Check if we have meaningful GPU memory data (CUDA only)
    has_gpu_memory = not all(r["memory_after_mb"]["allocated_mb"] == 0.0 for r in eval_results)
    
    if has_gpu_memory:
        eval_runtime_memory = [r["memory_after_mb"]["allocated_mb"] for r in eval_results]
        deploy_runtime_memory = [r["memory_after_mb"]["allocated_mb"] for r in deploy_results]
        eval_peak = [r["memory_after_mb"]["max_allocated_mb"] for r in eval_results]
        deploy_peak = [r["memory_after_mb"]["max_allocated_mb"] for r in deploy_results]
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        # Create figure with one subplot for static memory only
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        ax2 = None
        ax3 = None
    
    # Plot 1: Static Model Memory (always shown)
    ax1.plot(batch_sizes, eval_model_memory, marker='o', linewidth=2, 
             markersize=6, label='Native Pytorch', color='#3498db')
    ax1.plot(batch_sizes, deploy_model_memory, marker='s', linewidth=2, 
             markersize=6, label='Packed BitLinear', color='#e74c3c')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Layer Memory (MB)', fontsize=12)
    ax1.set_title('Static Layer Memory', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    # Set explicit ticks to show actual batch size values
    batch_size_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ax1.set_xticks(batch_size_ticks)
    ax1.set_xticklabels([str(bs) for bs in batch_size_ticks])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Add horizontal lines showing the constant model memory
    if len(eval_model_memory) > 0:
        ax1.axhline(y=eval_model_memory[0], color='#3498db', linestyle='--', 
                   linewidth=1, alpha=0.5)
        ax1.axhline(y=deploy_model_memory[0], color='#e74c3c', linestyle='--', 
                   linewidth=1, alpha=0.5)
    
    if has_gpu_memory:
        # Plot 2: Runtime Allocated Memory
        ax2.plot(batch_sizes, eval_runtime_memory, marker='o', linewidth=2, 
                 markersize=6, label='Native Pytorch', color='#3498db')
        ax2.plot(batch_sizes, deploy_runtime_memory, marker='s', linewidth=2, 
                 markersize=6, label='Packed BitLinear', color='#e74c3c')
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Runtime Memory (MB)', fontsize=12)
        ax2.set_title('GPU Runtime Memory (Allocated)', fontsize=14, fontweight='bold')
        ax2.set_xscale('log', base=2)
        # Set explicit ticks to show actual batch size values
        batch_size_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        ax2.set_xticks(batch_size_ticks)
        ax2.set_xticklabels([str(bs) for bs in batch_size_ticks])
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)
        
        # Plot 3: Peak Memory
        ax3.plot(batch_sizes, eval_peak, marker='o', linewidth=2, 
                 markersize=6, label='Native Pytorch Peak', color='#3498db')
        ax3.plot(batch_sizes, deploy_peak, marker='s', linewidth=2, 
                 markersize=6, label='Packed BitLinear Peak', color='#e74c3c')
        ax3.set_xlabel('Batch Size', fontsize=12)
        ax3.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax3.set_title('GPU Peak Memory Usage', fontsize=14, fontweight='bold')
        ax3.set_xscale('log', base=2)
        # Set explicit ticks to show actual batch size values
        ax3.set_xticks(batch_size_ticks)
        ax3.set_xticklabels([str(bs) for bs in batch_size_ticks])
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=11)
        
        print(f"✓ Memory plot saved to {output_path}")
    else:
        print(f"✓ Static memory plot saved to {output_path} (GPU memory data not available)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_static_memory(all_results: list, output_path: str, labels: list = None) -> None:
    """Plot static model memory consumption (weights and buffers only),
    and latency speedup next to it.
    """
    # If all_results is a single list (old format), wrap it
    if not isinstance(all_results[0], list):
        all_results = [all_results]
        labels = labels or ["Device"]
    
    # Use the first device's results for static memory (memory is the same across devices)
    results = all_results[0]
    
    # Separate eval and deploy results
    eval_results = [r for r in results if r["mode"] == "eval"]
    deploy_results = [r for r in results if r["mode"] == "deploy"]
    
    # Get unique model memory values (should be constant across batch sizes)
    eval_model_memory = eval_results[0]["model_memory_mb"] if eval_results else 0
    deploy_model_memory = deploy_results[0]["model_memory_mb"] if deploy_results else 0
    
    # Calculate memory savings
    memory_reduction = eval_model_memory - deploy_model_memory
    memory_reduction_pct = (memory_reduction / eval_model_memory * 100) if eval_model_memory > 0 else 0

    # Check if we can plot speedup for any device
    can_plot_speedup = any(
        bool([r for r in res if r["mode"] == "eval"]) and 
        bool([r for r in res if r["mode"] == "deploy"])
        for res in all_results
    )

    if can_plot_speedup:
        # Two subplots: memory bar + speedup
        fig, (ax_mem, ax_speedup) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        # Fallback to just memory if something is missing
        fig, ax_mem = plt.subplots(1, 1, figsize=(8, 6))
        ax_speedup = None
    
    # ----------------- Subplot 1: Static memory bar chart -----------------
    modes = ['Native Pytorch', 'Packed BitLinear']
    memories = [eval_model_memory, deploy_model_memory]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax_mem.bar(modes, memories, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, memory in zip(bars, memories):
        height = bar.get_height()
        ax_mem.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{memory:.2f} MB',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax_mem.set_ylabel('Layer Memory (MB)', fontsize=13, fontweight='bold')
    
    # Add reduction info to title
    if memory_reduction > 0:
        title_text = (
            f'Static Layer Memory Consumption\n'
            f'{memory_reduction_pct:.1f}% reduction: '
            f'{eval_model_memory:.2f} MB → {deploy_model_memory:.2f} MB'
        )
    else:
        title_text = 'Static Layer Memory Consumption'
    
    ax_mem.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    ax_mem.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax_mem.set_axisbelow(True)

    # ----------------- Subplot 2: Speedup vs batch size -----------------
    if can_plot_speedup:
        # Define color schemes and markers matching latency_comparison
        color_schemes = [
            '#3498db',  # Blue (default)
            '#2ecc71',  # Green
            '#9b59b6',  # Purple
            '#1abc9c',  # Teal
        ]
        markers = ['o', 's', 'D', '^']
        
        # Plot speedup for each device
        for idx, (results, label) in enumerate(zip(all_results, labels)):
            # Separate eval and deploy results for this device
            eval_results = [r for r in results if r["mode"] == "eval"]
            deploy_results = [r for r in results if r["mode"] == "deploy"]
            
            if not eval_results or not deploy_results:
                continue
            
            # Align eval/deploy entries by batch size (assumes same set/order)
            batch_sizes = [r["batch_size"] for r in eval_results]
            eval_latencies = [r["latency_ms"] for r in eval_results]
            deploy_latencies = [r["latency_ms"] for r in deploy_results]
            
            speedup = [e / d for e, d in zip(eval_latencies, deploy_latencies)]
            
            # Format label to match latency_comparison
            if label == "CPU": 
                label_str = "CPU (x86)"
            elif label == "CUDA": 
                label_str = "CUDA (5090)"
            else: 
                label_str = label
            
            color = color_schemes[idx % len(color_schemes)]
            marker = markers[idx % len(markers)]
            
            ax_speedup.plot(batch_sizes, speedup,
                            marker=marker, linewidth=2, markersize=6,
                            label=label_str, color=color, alpha=0.8)
            
            # Add text annotation for CUDA line showing speedup value
            if label == "CUDA" and len(speedup) > 0:
                # Use the first point for annotation
                annot_idx = 0
                annot_x = batch_sizes[annot_idx]
                annot_y = speedup[annot_idx]
                # Format speedup value (e.g., "1.2345× speedup")
                speedup_text = f"{annot_y:.4f}× speedup"
                ax_speedup.annotate(speedup_text,
                                   xy=(annot_x, annot_y),
                                   xytext=(20, 20), textcoords='offset points',
                                   fontsize=10, fontweight='bold',
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            edgecolor=color, alpha=0.8),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                                  color=color, lw=1.5))
        
        ax_speedup.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_speedup.set_xscale('log', base=2)
        # Set explicit ticks to show actual batch size values
        batch_size_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        ax_speedup.set_xticks(batch_size_ticks)
        ax_speedup.set_xticklabels([str(bs) for bs in batch_size_ticks])
        ax_speedup.set_xlabel('Batch Size', fontsize=12)
        ax_speedup.set_ylabel('Speedup (Native Pytorch / Packed BitLinear)', fontsize=12)
        ax_speedup.set_title('Latency Speedup vs Batch Size', fontsize=14, fontweight='bold')
        ax_speedup.grid(True, alpha=0.3, linestyle='--')
        ax_speedup.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Static memory + speedup plot saved to {output_path}")
    print(f"  Native Pytorch: {eval_model_memory:.2f} MB")
    print(f"  Packed BitLinear: {deploy_model_memory:.2f} MB")
    print(f"  Reduction: {memory_reduction:.2f} MB ({memory_reduction_pct:.1f}%)")
    plt.close()


def create_summary_table(results: list, output_path: str) -> None:
    """Create a summary table image showing key metrics."""
    eval_results = [r for r in results if r["mode"] == "eval"]
    deploy_results = [r for r in results if r["mode"] == "deploy"]
    
    # Select a few key batch sizes for the table
    key_batch_sizes = [1, 32, 128, 1024]
    table_data = []
    
    for bs in key_batch_sizes:
        eval_r = next((r for r in eval_results if r["batch_size"] == bs), None)
        deploy_r = next((r for r in deploy_results if r["batch_size"] == bs), None)
        
        if eval_r and deploy_r:
            speedup = eval_r["latency_ms"] / deploy_r["latency_ms"]
            table_data.append([
                bs,
                f"{eval_r['latency_ms']:.2f}",
                f"{deploy_r['latency_ms']:.2f}",
                f"{speedup:.2f}x",
                f"{deploy_r['gflops']:.1f}"
            ])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Batch Size', 'Eval Latency (ms)', 'Deploy Latency (ms)', 'Speedup', 'GFLOP/s']
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.22, 0.22, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary table saved to {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BitLinear profiling results.")
    parser.add_argument(
        "--input",
        nargs='+',
        default=["profile_results.json"],
        help="Input JSON file(s) with profiling results. Can provide multiple files for device comparison (default: profile_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--labels",
        nargs='+',
        default=None,
        help="Labels for each input file (e.g., 'CPU' 'CUDA'). If not provided, uses device names from files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Ensure input is a list
    input_files = args.input if isinstance(args.input, list) else [args.input]
    
    # Load results from all files
    all_data = []
    all_results = []
    for input_file in input_files:
        data = load_results(input_file)
        all_data.append(data)
        all_results.append(data["results"])
    
    # Generate labels if not provided
    if args.labels:
        labels = args.labels
        if len(labels) != len(input_files):
            print(f"Warning: Number of labels ({len(labels)}) doesn't match number of input files ({len(input_files)})")
            print("Using device names from files instead")
            labels = None
    
    if labels is None:
        # Extract device names from data
        labels = []
        for data in all_data:
            deploy_device = data.get('deploy_device', data.get('train_device', 'Unknown'))
            # Simplify device name (e.g., "cuda:0" -> "CUDA", "cpu" -> "CPU")
            if 'cuda' in deploy_device.lower():
                device_label = 'CUDA'
            elif 'cpu' in deploy_device.lower():
                device_label = 'CPU'
            elif 'mps' in deploy_device.lower():
                device_label = 'MPS'
            else:
                device_label = deploy_device
            labels.append(device_label)
    
    # Print summary
    print(f"Loaded profiling results from {len(input_files)} file(s)")
    for label, data in zip(labels, all_data):
        print(f"  {label}: {data['train_device']} (train) -> {data['deploy_device']} (deploy)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_latency_comparison(all_results, str(output_dir / "latency_comparison.png"), labels)
    plot_throughput_comparison(all_results, str(output_dir / "throughput_comparison.png"), labels)
    plot_static_memory(all_results, str(output_dir / "static_memory.png"), labels)
    
    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()