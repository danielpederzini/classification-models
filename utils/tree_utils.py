import matplotlib.pyplot as plt

def assign_positions(node, positions, x, y, width, depth, max_depth=4):
    positions[node] = (x, y)
    if node.left:
        left_x = x - width / 3
        left_y = y - 0.2
        assign_positions(node.left, positions, left_x, left_y, width / 2, depth + 1, max_depth)
    if node.right:
        right_x = x + width / 3
        right_y = y - 0.2
        assign_positions(node.right, positions, right_x, right_y, width / 2, depth + 1, max_depth)
    
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
    width = max_len * 0.008 + 0.02
    height = len(lines) * 0.025 + 0.01
    
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
    def plot_decision_tree(tree, class_names, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.axis('off')
        
        positions = {}
        assign_positions(tree, positions, x=0.5, y=0.9, width=1.0, depth=0)
        draw_tree(ax, tree, positions, class_names, x=0.5, y=0.9, parent_x=None, parent_y=None)
        plt.show()