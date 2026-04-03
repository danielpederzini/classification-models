import matplotlib.pyplot as plt

def assign_positions(node, positions, depth=0, x_offset=[0], x_spacing=0.16, y_spacing=0.14):
    # Inorder assignment gives unique x positions for each node and avoids overlap in deep trees.
    if node.left:
        assign_positions(node.left, positions, depth + 1, x_offset, x_spacing, y_spacing)

    x = x_offset[0]
    y = 1.0 - depth * y_spacing
    positions[node] = (x, y)
    x_offset[0] += x_spacing

    if node.right:
        assign_positions(node.right, positions, depth + 1, x_offset, x_spacing, y_spacing)

    
def draw_tree(ax, node, positions, class_names, x, y, parent_x, parent_y):
    if parent_x is not None and parent_y is not None:
        ax.plot([parent_x, x], [parent_y, y], 'k-', lw=1, zorder=1)
    
    is_leaf = node.value is not None
    if is_leaf:
        prob = node.value
        label = f"{class_names[1] if prob >= 0.5 else class_names[0]}\n{prob:.2f}"
    else:
        label = f"{node.feature}\n<= {node.threshold:.2f}"
    
    lines = label.split('\n')
    max_len = max(len(line) for line in lines)
    width = max(0.12, max_len * 0.02)
    height = max(0.04, len(lines) * 0.015)
    
    facecolor = 'lightgreen' if is_leaf else 'lightyellow'
    rectangle = plt.Rectangle((x - width/2, y - height/2), width, height, facecolor=facecolor, edgecolor='black', zorder=3)
    ax.add_patch(rectangle)
    
    ax.text(x, y, label, ha='center', va='center', fontsize=8, zorder=4)
    
    if node.left:
        left_x, left_y = positions[node.left]
        draw_tree(ax, node.left, positions, class_names, left_x, left_y, x, y)
    if node.right:
        right_x, right_y = positions[node.right]
        draw_tree(ax, node.right, positions, class_names, right_x, right_y, x, y)
        

class TreeHelper():
    def plot_decision_tree(tree, class_names, figsize=(16, 10)):
        fig, ax = plt.subplots(figsize=figsize)

        positions = {}
        assign_positions(tree, positions)

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin = 0.1
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        ax.axis('off')
        root_x, root_y = positions[tree]
        draw_tree(ax, tree, positions, class_names, x=root_x, y=root_y, parent_x=None, parent_y=None)
        plt.show()

    def plot_forest(trees, n_plots=None, class_names=None, figsize=(18, 12)):
        if n_plots is None:
            n_plots = len(trees)
        n_plots = min(n_plots, len(trees))

        cols = min(2, n_plots) if n_plots > 0 else 1
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if isinstance(axes, plt.Axes):
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, tree in enumerate(trees[:n_plots]):
            ax = axes[i]
            positions = {}
            assign_positions(tree, positions)

            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            margin = 0.1
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
            ax.axis('off')

            root_x, root_y = positions[tree]
            draw_tree(ax, tree, positions, class_names or ['0', '1'], x=root_x, y=root_y, parent_x=None, parent_y=None)

        for j in range(n_plots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()