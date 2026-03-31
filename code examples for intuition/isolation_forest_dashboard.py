import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

np.random.seed(42)

# Generate 2D data: normal cluster 
n_normal = 60
x_normal = np.random.normal(15, 3, n_normal)
y_normal = np.random.normal(15, 3, n_normal)

# Outliers
outliers_x = np.array([2, 28, 3, 27, 5])
outliers_y = np.array([27, 3, 5, 26, 28])

X = np.column_stack([np.concatenate([x_normal, outliers_x]),
                      np.concatenate([y_normal, outliers_y])])
is_outlier = np.array([False]*n_normal + [True]*len(outliers_x))

# Build one isolation tree manually with recorded splits
class IsoNode:
    def __init__(self, data_idx, depth, bounds):
        self.data_idx = data_idx
        self.depth = depth
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = len(data_idx) <= 1 or depth >= 10

    def split(self, X):
        if self.is_leaf:
            return
        self.feature = np.random.randint(0, 2)
        vals = X[self.data_idx, self.feature]
        lo, hi = vals.min(), vals.max()
        if lo == hi:
            self.is_leaf = True
            return
        self.threshold = np.random.uniform(lo, hi)

        left_mask = X[self.data_idx, self.feature] < self.threshold
        left_idx = self.data_idx[left_mask]
        right_idx = self.data_idx[~left_mask]

        xmin, xmax, ymin, ymax = self.bounds
        if self.feature == 0:
            left_bounds = (xmin, self.threshold, ymin, ymax)
            right_bounds = (self.threshold, xmax, ymin, ymax)
        else:
            left_bounds = (xmin, xmax, ymin, self.threshold)
            right_bounds = (xmin, xmax, self.threshold, ymax)

        self.left = IsoNode(left_idx, self.depth + 1, left_bounds)
        self.right = IsoNode(right_idx, self.depth + 1, right_bounds)
        self.left.split(X)
        self.right.split(X)

def get_splits_up_to_depth(node, max_depth):
    """Get all split lines up to a certain depth"""
    splits = []
    if node is None or node.is_leaf or node.depth >= max_depth:
        return splits
    xmin, xmax, ymin, ymax = node.bounds
    if node.feature == 0:
        splits.append(('v', node.threshold, ymin, ymax, node.depth))
    else:
        splits.append(('h', node.threshold, xmin, xmax, node.depth))
    splits += get_splits_up_to_depth(node.left, max_depth)
    splits += get_splits_up_to_depth(node.right, max_depth)
    return splits

def get_leaf_info(node, max_depth):
    """Get leaf rectangles and their point counts at given depth"""
    leaves = []
    if node is None:
        return leaves
    if node.is_leaf or node.depth >= max_depth:
        leaves.append((node.bounds, len(node.data_idx), node.data_idx))
        return leaves
    leaves += get_leaf_info(node.left, max_depth)
    leaves += get_leaf_info(node.right, max_depth)
    return leaves

def get_path_length(node, point_idx, max_depth=99):
    """Get depth at which a specific point is isolated"""
    if node is None:
        return 0
    if len(node.data_idx) <= 1 or node.is_leaf:
        return node.depth
    if node.depth >= max_depth:
        return node.depth
    if point_idx in node.left.data_idx:
        return get_path_length(node.left, point_idx, max_depth)
    else:
        return get_path_length(node.right, point_idx, max_depth)

# Build tree
np.random.seed(15)  # seed for a nice-looking tree
bounds = (0, 30, 0, 30)
root = IsoNode(np.arange(len(X)), 0, bounds)
root.split(X)

# Get path lengths for all points
path_lengths = np.array([get_path_length(root, i) for i in range(len(X))])

# Interactive plot
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
gs.update(bottom=0.18)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

depth_colors = plt.cm.Set1(np.linspace(0, 1, 10))

def update(val):
    d = int(slider.val)
    ax1.clear()
    ax2.clear()

    # Left plot: scatter + splits
    splits = get_splits_up_to_depth(root, d)
    leaves = get_leaf_info(root, d)

    # Color leaves by number of points
    for bounds_rect, count, idx in leaves:
        xmin, xmax, ymin, ymax = bounds_rect
        if count <= 1:
            color = '#ffcdd2'
            alpha = 0.35
        elif count <= 3:
            color = '#fff9c4'
            alpha = 0.25
        else:
            color = '#c8e6c9'
            alpha = 0.2
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          facecolor=color, edgecolor='none', alpha=alpha, zorder=1)
        ax1.add_patch(rect)

    # Draw splits
    for s in splits:
        lw = max(0.8, 2.5 - s[4] * 0.3)
        alpha = max(0.3, 1.0 - s[4] * 0.1)
        col = depth_colors[s[4] % 10]
        if s[0] == 'v':
            ax1.plot([s[1], s[1]], [s[2], s[3]], '-', color=col, lw=lw, alpha=alpha, zorder=2)
        else:
            ax1.plot([s[2], s[3]], [s[1], s[1]], '-', color=col, lw=lw, alpha=alpha, zorder=2)

    # Points
    ax1.scatter(X[~is_outlier, 0], X[~is_outlier, 1], c='#1565c0', s=40, alpha=0.7,
                edgecolors='#333', linewidths=0.4, zorder=3, label='Normal')
    ax1.scatter(X[is_outlier, 0], X[is_outlier, 1], c='#c62828', s=80, alpha=0.9,
                edgecolors='#333', linewidths=0.8, zorder=4, marker='*', label='Anomaly')

    # Mark isolated points
    for bounds_rect, count, idx in leaves:
        if count == 1:
            px, py = X[idx[0]]
            ax1.scatter(px, py, facecolors='none', edgecolors='#ff6f00',
                        s=250, linewidths=2.5, zorder=5)

    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 30)
    ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax1.set_title(f'Isolation Tree — Depth {d}\n'
                  f'(orange circle = isolated)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_aspect('equal')

    # Right plot: path length bar chart
    # Show a few selected points
    selected_normal = [0, 5, 10, 20, 30]
    selected_outlier = list(range(n_normal, n_normal + len(outliers_x)))
    selected = selected_normal + selected_outlier

    names = [f'Normal {i+1}' for i in range(len(selected_normal))] + \
            [f'Anomaly {i+1}' for i in range(len(selected_outlier))]
    
    current_depths = []
    for idx in selected:
        pl = get_path_length(root, idx, max_depth=d)
        current_depths.append(pl)

    colors = ['#1565c0'] * len(selected_normal) + ['#c62828'] * len(selected_outlier)
    bars = ax2.barh(range(len(selected)), current_depths, color=colors, 
                     edgecolor='#333', linewidth=0.5, alpha=0.8)
    ax2.set_yticks(range(len(selected)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Path Length (splits to isolate)', fontsize=11, fontweight='bold')
    ax2.set_title('Path Lengths at Current Depth', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.grid(True, axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, current_depths):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val}', va='center', fontsize=9, fontweight='bold')

    fig.canvas.draw_idle()

# Slider
ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03])
slider = Slider(ax_slider, 'Tree Depth', 1, 8, valinit=1, valstep=1,
                color='#1565c0')
slider.on_changed(update)

fig.suptitle('Isolation Forest: How Anomalies Get Isolated Faster',
             fontsize=15, fontweight='bold', y=0.98)

update(1)
plt.show()
