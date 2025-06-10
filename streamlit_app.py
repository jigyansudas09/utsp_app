import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, GINConv, SAGEConv
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import os
import tempfile
import elkai

# Initialize session state variables
if 'training_state' not in st.session_state:
    st.session_state.training_state = {
        'is_training': False,
        'current_epoch': 0,
        'total_epochs': 0,
        'current_batch': 0,
        'total_batches': 0,
        'model_state': None,
        'optimizer_state': None,
        'metrics_history': {},
        'current_loss': 0.0,
        'best_tour_length': float('inf'),
        'best_model_state': None
    }

# Initialize experiment results state
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = {}

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def safe_to_numpy(tensor):
    """Safely converts a PyTorch tensor to a NumPy array."""
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor

class TSPLIB_Parser:
    """A self-contained utility to load the berlin52 TSP instance."""
    def get_berlin52(self):
        coords = np.array([
            [565.0, 575.0],[25.0, 185.0],[345.0, 750.0],[945.0, 685.0],[845.0, 655.0],[880.0, 660.0],[25.0, 230.0],
            [525.0, 1000.0],[580.0, 1175.0],[650.0, 1130.0],[1605.0, 620.0],[1220.0, 580.0],[1465.0, 200.0],
            [1530.0, 5.0],[845.0, 680.0],[725.0, 370.0],[145.0, 665.0],[415.0, 635.0],[510.0, 875.0],
            [560.0, 365.0],[300.0, 465.0],[520.0, 585.0],[480.0, 415.0],[835.0, 625.0],[975.0, 580.0],
            [1215.0, 245.0],[1320.0, 315.0],[1250.0, 400.0],[660.0, 180.0],[410.0, 250.0],[420.0, 555.0],
            [575.0, 665.0],[1150.0, 1160.0],[700.0, 580.0],[685.0, 595.0],[685.0, 610.0],[770.0, 610.0],
            [795.0, 645.0],[720.0, 635.0],[760.0, 650.0],[475.0, 960.0],[95.0, 260.0],[875.0, 920.0],
            [700.0, 500.0],[555.0, 815.0],[830.0, 485.0],[1170.0, 65.0],[830.0, 610.0],[605.0, 625.0],
            [595.0, 360.0],[1340.0, 725.0],[1740.0, 245.0]
        ])
        return coords, 7542.0

class TSPDataset(Dataset):
    def __init__(self, num_nodes=50, num_samples=1000, seed=42, coords=None):
        if coords is not None:
            self.instances = [coords] * num_samples
        else:
            np.random.seed(seed)
            self.instances = [np.random.uniform(0, 1, (num_nodes, 2)) for _ in range(num_samples)]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        coords = torch.tensor(self.instances[idx], dtype=torch.float)
        return Data(
            x=coords,
            pos=coords,
            coords=coords,
            distances=torch.cdist(coords, coords),
            num_nodes=coords.shape[0]
        )

class ScatteringAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.W_q = nn.Linear(in_features, out_features * n_heads)
        self.W_k = nn.Linear(in_features, out_features * n_heads)
        self.W_v = nn.Linear(in_features, out_features * n_heads)
        self.W_psi = nn.Linear(out_features, out_features)
        self.aggregator = torch_geometric.nn.aggr.SumAggregation()
        self.output_layer = nn.Linear(out_features * n_heads * 2, in_features)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x, edge_index):
        N, E = x.size(0), edge_index.size(1)
        q, k, v = (w(x).view(N, self.n_heads, self.out_features) for w in [self.W_q, self.W_k, self.W_v])
        row, col = edge_index

        alpha = torch_geometric.utils.softmax((q[row] * k[col]).sum(dim=-1) / np.sqrt(self.out_features), row, num_nodes=N)

        v_j_lp = v[col]
        h_lp = self.aggregator((alpha.unsqueeze(-1) * v_j_lp).view(E, -1), row, dim_size=N)

        v_j_transformed = torch.tanh(self.W_psi(v[col].view(-1, self.out_features)))
        v_j_bp = v_j_transformed.view(E, self.n_heads, self.out_features)
        h_bp = self.aggregator((alpha.unsqueeze(-1) * v_j_bp).view(E, -1), row, dim_size=N)

        h_out = self.output_layer(torch.cat([h_lp, h_bp], dim=1))
        return self.layer_norm(x + F.relu(h_out))

class GNNBackbone(nn.Module):
    def __init__(self, gnn_type, node_dim, hidden_dim, n_layers, n_heads=8):
        super().__init__()
        self.gnn_type = gnn_type
        self.embedding = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList()
        in_dim = hidden_dim

        for i in range(n_layers):
            if self.gnn_type == 'SAG':
                self.layers.append(ScatteringAttentionLayer(in_dim, hidden_dim // n_heads, n_heads))
            elif self.gnn_type == 'GAT':
                self.layers.append(GATv2Conv(in_dim, hidden_dim, heads=n_heads, concat=True))
                in_dim = hidden_dim * n_heads
            elif self.gnn_type == 'GIN':
                mlp = nn.Sequential(nn.Linear(in_dim, 2*in_dim), nn.ReLU(), nn.Linear(2*in_dim, in_dim))
                self.layers.append(GINConv(mlp))
            elif self.gnn_type == 'GraphSAGE':
                self.layers.append(SAGEConv(in_dim, in_dim))

        self.output_head = nn.Linear(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.embedding(x))
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        return self.output_head(h)

class TSPModel(nn.Module):
    def __init__(self, gnn_type, loss_type, node_dim=2, hidden_dim=64, n_layers=3, n_heads=8):
        super().__init__()
        self.loss_type = loss_type
        self.gnn = GNNBackbone(gnn_type, node_dim, hidden_dim, n_layers, n_heads)

        if self.loss_type == 'EdgeHeatmapLoss':
            self.edge_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, data):
        if self.loss_type == 'UTSPLoss':
            edge_index = to_undirected(torch.combinations(torch.arange(data.num_nodes, device=data.x.device), r=2).t())
        else:
            edge_index = data.edge_index

        h = self.gnn(data.x, edge_index)

        if self.loss_type == 'UTSPLoss':
            return h @ h.T
        else:
            row, col = edge_index
            edge_features = torch.cat([h[row], h[col]], dim=1)
            return self.edge_predictor(edge_features).squeeze(-1)

class UTSPLoss(nn.Module):
    def __init__(self, temperature=0.1, c1_penalty=1.0, c2_penalty=1.0):
        super().__init__()
        self.temperature = temperature
        self.c1_penalty = c1_penalty
        self.c2_penalty = c2_penalty

    def forward(self, S, data):
        T = F.softmax(S / self.temperature, dim=0)
        H = T @ torch.cat([T[:, -1].unsqueeze(1), T[:, :-1]], dim=1).T
        loss_tsp = (H * data.distances).sum()
        loss_c1 = ((T.sum(dim=1) - 1.0) ** 2).mean()
        loss_c2 = ((T.sum(dim=0) - 1.0) ** 2).mean()
        return loss_tsp + self.c1_penalty * loss_c1 + self.c2_penalty * loss_c2

class EdgeHeatmapLoss(nn.Module):
    def __init__(self, degree_penalty_weight=2.0):
        super().__init__()
        self.lambda_deg = degree_penalty_weight

    def forward(self, edge_logits, data):
        edge_probs = torch.sigmoid(edge_logits)
        edge_distances = data.distances[data.edge_index[0], data.edge_index[1]]
        loss_dist = (edge_probs * edge_distances).sum()
        weighted_degree = torch_geometric.utils.scatter(
            edge_probs, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce='sum'
        )
        loss_deg = ((weighted_degree - 2.0) ** 2).mean()
        return loss_dist + self.lambda_deg * loss_deg

class TSPSolver:
    def calculate_tour_length(self, coords, tour):
        coords_np = safe_to_numpy(coords)
        if not tour or len(tour) != coords_np.shape[0]:
            return float('inf')
        return np.linalg.norm(coords_np[tour] - np.roll(coords_np[tour], -1, axis=0), axis=1).sum()

    def model_solve(self, coords, heatmap, edge_index=None):
        initial_tour = self._greedy_from_heatmap(coords.shape[0], heatmap, edge_index)
        return self._solve_2_opt(initial_tour, safe_to_numpy(coords))

    def baseline_solve_nn(self, coords):
        return self.nearest_neighbor_tour(safe_to_numpy(coords))

    def baseline_solve_elkai(self, coords):
        coords_np = safe_to_numpy(coords)
        N = coords_np.shape[0]
        
        try:
            scaling_factor = 10000
            cities_dict = {i: (int(c[0] * scaling_factor), int(c[1] * scaling_factor)) for i, c in enumerate(coords_np)}
            problem = elkai.Coordinates2D(cities_dict)
            elkai_tour = problem.solve_tsp()
            
            if len(elkai_tour) == N + 1:
                tour = elkai_tour[:-1]
            else:
                tour = elkai_tour
            
            length = self.calculate_tour_length(coords, tour)
            return tour, length
            
        except Exception as e:
            st.error(f"Elkai solver failed: {e}")
            nn_tour = self.nearest_neighbor_tour(coords_np)
            opt_tour = self._solve_2_opt(nn_tour, coords_np)
            return opt_tour, self.calculate_tour_length(coords, opt_tour)

    def nearest_neighbor_tour(self, coords):
        N = len(coords)
        tour = [0]
        visited = {0}
        while len(visited) < N:
            last = tour[-1]
            next_node = min((i for i in range(N) if i not in visited), 
                          key=lambda x: np.linalg.norm(coords[last] - coords[x]))
            tour.append(next_node)
            visited.add(next_node)
        return tour

    def _greedy_from_heatmap(self, N, heatmap, edge_index):
        if heatmap.ndim == 2:
            H = safe_to_numpy(heatmap)
            tour = [0]
            visited = {0}
            current = 0
            while len(visited) < N:
                candidates = sorted([(i, H[current, i]) for i in range(N) if i not in visited], 
                                 key=lambda x: x[1], reverse=True)
                if not candidates:
                    break
                tour.append(candidates[0][0])
                visited.add(candidates[0][0])
                current = candidates[0][0]
            return tour
        else:
            adj = {i: [] for i in range(N)}
            edge_probs = safe_to_numpy(heatmap)
            if edge_index is not None:
                for i, (u, v) in enumerate(edge_index.T):
                    adj[u.item()].append((v.item(), edge_probs[i]))
            tour = [0]
            visited = {0}
            current = 0
            while len(visited) < N:
                neighbors = sorted([n for n in adj.get(current, []) if n[0] not in visited], 
                                key=lambda x: x[1], reverse=True)
                if not neighbors:
                    unvisited = list(set(range(N)) - visited)
                    next_node = unvisited[0] if unvisited else -1
                else:
                    next_node = neighbors[0][0]
                if next_node == -1:
                    break
                tour.append(next_node)
                visited.add(next_node)
                current = next_node
            return tour

    def _solve_2_opt(self, tour, coords):
        if len(tour) < 4:
            return tour
        num_nodes = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(1, num_nodes - 2):
                for j in range(i + 1, num_nodes):
                    if j-i == 1:
                        continue
                    old_dist = (np.linalg.norm(coords[tour[i-1]]-coords[tour[i]]) + 
                              np.linalg.norm(coords[tour[j]]-coords[tour[(j+1)%num_nodes]]))
                    new_dist = (np.linalg.norm(coords[tour[i-1]]-coords[tour[j]]) + 
                              np.linalg.norm(coords[tour[i]]-coords[tour[(j+1)%num_nodes]]))
                    if new_dist < old_dist:
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        return tour

class EnhancedVisualizer:
    def __init__(self):
        self.metrics_history = {}
        self.status_container = st.container()

    def update_metrics(self, model_key, epoch, metrics_dict):
        if model_key not in self.metrics_history:
            self.metrics_history[model_key] = []
        self.metrics_history[model_key].append({'epoch': epoch, **metrics_dict})

    def plot_training_progress(self, model_key):
        if not self.metrics_history.get(model_key):
            return
        
        df = pd.DataFrame(self.metrics_history[model_key])
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Live Training Progress: {model_key}', fontsize=18, fontweight='bold')
        
        axes[0, 0].plot(df['epoch'], df['loss'], 'b-', label='Avg. Training Loss')
        axes[0, 0].set_title('Training Loss')
        
        axes[0, 1].plot(df['epoch'], df['utsp_tour_length'], 'g-', label='UTSP (2-Opt) Length')
        axes[0, 1].plot(df['epoch'], df['elkai_tour_length'], 'purple', linestyle='-.', label='Elkai Solver Length')
        axes[0, 1].plot(df['epoch'], df['nn_tour_length'], 'r--', label='Nearest Neighbor Length')
        axes[0, 1].set_title('Validation Tour Lengths')
        
        if 'optimality_gap' in df.columns and not df['optimality_gap'].isnull().all():
            axes[1, 0].plot(df['epoch'], df['optimality_gap'], 'm', label='Optimality Gap (%)')
            axes[1, 0].set_title('Optimality Gap')
        else:
            axes[1, 0].axis('off')
            axes[1, 0].set_title('No Optimal Length Provided')
        
        axes[1, 1].plot(df['epoch'], df['improvement_vs_nn'], 'orange', label='% Improvement vs. NN')
        axes[1, 1].set_title('Improvement vs. Baseline')
        axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.4)
            ax.set_xlabel('Epoch')
            ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)
        plt.close(fig)

    def plot_tour_comparison(self, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Final Tour Comparison', fontsize=18, fontweight='bold')
        
        self.plot_heatmap(axes[0, 0], metrics['coords'], metrics['heatmap'], metrics['edge_index'])
        self.plot_tour(axes[0, 1], metrics['coords'], metrics['utsp_tour'], 
                      f"Your Model's Tour (2-Opt)\nLength: {metrics['utsp_tour_length']:.3f}")
        self.plot_tour(axes[1, 0], metrics['coords'], metrics['nn_tour'], 
                      f"Nearest Neighbor Baseline\nLength: {metrics['nn_tour_length']:.3f}")
        self.plot_tour(axes[1, 1], metrics['coords'], metrics['elkai_tour'], 
                      f"Elkai Solver Tour\nLength: {metrics['elkai_tour_length']:.3f}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)

    def plot_heatmap(self, ax, coords, heatmap, edge_index=None):
        coords_np = safe_to_numpy(coords)
        
        if edge_index is not None:
            # For EdgeHeatmapLoss: Plot edge probabilities
            edge_probs = safe_to_numpy(heatmap)
            
            # Plot nodes in gray/black for better contrast
            ax.scatter(coords_np[:, 0], coords_np[:, 1], c='black', s=60, zorder=3)
            
            # Convert edge_index to numpy if it's a tensor
            edge_index_np = safe_to_numpy(edge_index)
            
            # Normalize probabilities for better visualization
            if len(edge_probs) > 0:
                min_prob, max_prob = edge_probs.min(), edge_probs.max()
                if max_prob > min_prob:
                    normalized_probs = (edge_probs - min_prob) / (max_prob - min_prob)
                else:
                    normalized_probs = edge_probs
            else:
                normalized_probs = edge_probs
            
            # Plot edges with probabilities using colormap
            import matplotlib.cm as cm
            cmap = cm.get_cmap('viridis')  # or 'plasma', 'hot', 'cool'
            
            for i in range(edge_index_np.shape[1]):
                u, v = edge_index_np[0, i], edge_index_np[1, i]
                prob = edge_probs[i]
                norm_prob = normalized_probs[i]
                
                # Lower threshold and better visibility
                if prob > 0.01:  # Much lower threshold
                    color = cmap(norm_prob)
                    # Use fixed alpha for visibility, vary linewidth instead
                    linewidth = 0.5 + 4 * norm_prob  # Range: 0.5 to 4.5
                    ax.plot([coords_np[u, 0], coords_np[v, 0]], 
                           [coords_np[u, 1], coords_np[v, 1]], 
                           color=color, 
                           alpha=0.7,  # Fixed alpha for visibility
                           linewidth=linewidth, 
                           zorder=1)
            
            ax.set_title('Edge Probability Heatmap')
            
            # Add colorbar for edge probabilities
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_prob, vmax=max_prob))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Edge Probability')
            
        else:
            # For UTSPLoss: Plot NxN heatmap
            heatmap_np = safe_to_numpy(heatmap)
            im = ax.imshow(heatmap_np, cmap='viridis')
            ax.set_title('UTSP Heatmap Matrix')
            plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

    def plot_tour(self, ax, coords, tour, title):
        coords_np = safe_to_numpy(coords)
        tour_np = np.array(tour)
        
        ax.scatter(coords_np[:, 0], coords_np[:, 1], c='blue', s=50)
        for i in range(len(tour_np)):
            start = coords_np[tour_np[i]]
            end = coords_np[tour_np[(i + 1) % len(tour_np)]]
            ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=1)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

    def plot_final_comparison(self, all_results, gold_standard_length=None):
        labels, tour_lengths = [], []
        for model_key, result in all_results.items():
            labels.append(model_key)
            tour_lengths.append(result['utsp_tour_length'])
        
        first_key = list(all_results.keys())[0]
        nn_length = all_results[first_key]['nn_tour_length']
        elkai_length = all_results[first_key]['elkai_tour_length']
        
        labels.extend(['Nearest Neighbor', 'Elkai Solver'])
        tour_lengths.extend([nn_length, elkai_length])
        
        if gold_standard_length:
            labels.append('Optimal Solution')
            tour_lengths.append(gold_standard_length)
        
        fig = plt.figure(figsize=(12, 7))
        bars = plt.bar(labels, tour_lengths, color=sns.color_palette("viridis", len(labels)))
        
        plt.ylabel('Final Tour Length', fontsize=12)
        plt.title('Final Model Performance Comparison', fontsize=18, fontweight='bold')
        plt.xticks(rotation=40, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{bar.get_height():.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

class CustomKNNGraph(BaseTransform):
    def __init__(self, k=20, loop=False):
        self.k = k
        self.loop = loop

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        
        # Calculate pairwise distances
        dist = torch.cdist(data.pos, data.pos)
        
        # Get k nearest neighbors for each node
        if not self.loop:
            dist.fill_diagonal_(float('inf'))
        
        # Get indices of k nearest neighbors
        _, col = torch.topk(dist, k=self.k, dim=1, largest=False)
        row = torch.arange(data.pos.size(0), device=data.pos.device).view(-1, 1).repeat(1, self.k)
        
        # Create edge index
        edge_index = torch.stack([row.view(-1), col.view(-1)], dim=0)
        
        # Make edges undirected
        edge_index = to_undirected(edge_index)
        
        data.edge_index = edge_index
        return data

class UTSPTrainer:
    def __init__(self, model, device, visualizer):
        self.model = model.to(device)
        self.device = device
        self.solver = TSPSolver()
        self.visualizer = visualizer
        self.status_container = st.container()

    def train(self, dataset, loss_type, batch_size, epochs, lr, optimal_length, visualize_every, model_key, **loss_params):
        if batch_size > 1:
            st.warning("Current loss implementations require batch_size=1 for stability. Overriding setting.")
            batch_size = 1

        if loss_type == 'UTSPLoss':
            dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
            criterion = UTSPLoss(**loss_params).to(self.device)
        else:
            dataloader = PyGDataLoader([CustomKNNGraph(k=20, loop=False)(d.clone()) for d in dataset],
                                     batch_size=batch_size, shuffle=True)
            criterion = EdgeHeatmapLoss(**loss_params).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize or restore training state
        if st.session_state.training_state['model_state'] is not None:
            try:
                # Try to load the state dict
                self.model.load_state_dict(st.session_state.training_state['model_state'])
                optimizer.load_state_dict(st.session_state.training_state['optimizer_state'])
                start_epoch = st.session_state.training_state['current_epoch']
            except RuntimeError as e:
                # If there's a mismatch in architecture, start fresh
                st.warning("Previous model state has different architecture. Starting fresh training.")
                st.session_state.training_state['model_state'] = None
                st.session_state.training_state['optimizer_state'] = None
                start_epoch = 0
        else:
            start_epoch = 0

        # Update session state
        st.session_state.training_state.update({
            'is_training': True,
            'total_epochs': epochs,
            'total_batches': len(dataloader),
            'current_epoch': start_epoch
        })

        # Create progress bars and status containers
        epoch_progress = st.progress(0)
        batch_progress = st.progress(0)
        status_text = st.empty()
        
        try:
            for epoch in range(start_epoch, epochs):
                self.model.train()
                total_loss = 0
                
                # Update epoch progress
                epoch_progress.progress((epoch + 1) / epochs)
                status_text.text(f'Training epoch {epoch+1}/{epochs}')
                
                for i, data in enumerate(dataloader):
                    # Update batch progress and session state
                    batch_progress.progress((i + 1) / len(dataloader))
                    st.session_state.training_state['current_batch'] = i
                    
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                    # Update current loss in session state
                    st.session_state.training_state['current_loss'] = total_loss / (i + 1)

                # Save model and optimizer state
                st.session_state.training_state.update({
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'current_epoch': epoch + 1
                })

                if (epoch + 1) % visualize_every == 0 or epoch == epochs - 1:
                    with self.status_container:
                        metrics = self.evaluate(dataset[0], total_loss / len(dataloader),
                                             optimal_length, model_key, epoch+1)
                        
                        # Update best model if current tour length is better
                        if metrics['utsp_tour_length'] < st.session_state.training_state['best_tour_length']:
                            st.session_state.training_state['best_tour_length'] = metrics['utsp_tour_length']
                            st.session_state.training_state['best_model_state'] = self.model.state_dict()
                        
                        self.visualizer.update_metrics(model_key, epoch + 1, metrics)
                        self.visualizer.plot_training_progress(model_key)

        except Exception as e:
            st.error(f"Training interrupted: {str(e)}")
            # Save current state for potential resume
            st.session_state.training_state['is_training'] = False
            return None

        finally:
            # Clear progress bars after training
            epoch_progress.empty()
            batch_progress.empty()
            status_text.empty()
            
            # Reset training state
            st.session_state.training_state['is_training'] = False

        # Load best model state before final evaluation
        if st.session_state.training_state['best_model_state'] is not None:
            self.model.load_state_dict(st.session_state.training_state['best_model_state'])

        with self.status_container:
            final_metrics = self.evaluate(dataset[0], total_loss / len(dataloader),
                                        optimal_length, model_key, epochs, is_final=True)
            self.visualizer.plot_tour_comparison(final_metrics)
        return final_metrics

    def evaluate(self, sample_data, avg_loss, optimal_length=None, model_key="", epoch=0, is_final=False):
        self.model.eval()
        with torch.no_grad():
            eval_data = sample_data.clone()
            
            heatmap_for_solver = None
            heatmap_for_viz = None
            edge_index_for_viz = None
            
            if self.model.loss_type == 'UTSPLoss':
                device_data = eval_data.to(self.device)
                output = self.model(device_data)
                
                T = F.softmax(output/0.1, dim=0)
                heatmap_for_solver = T @ torch.cat([T[:, -1].unsqueeze(1), T[:, :-1]], dim=1).T
                
                edge_index_for_viz = to_undirected(torch.combinations(
                    torch.arange(eval_data.num_nodes, device=output.device), r=2).t())
                heatmap_for_viz = heatmap_for_solver[edge_index_for_viz[0], edge_index_for_viz[1]]
                edge_index_for_solve = None
            else:
                knn_data = CustomKNNGraph(k=20, loop=False)(eval_data).to(self.device)
                output = self.model(knn_data)
                
                heatmap_prob = torch.sigmoid(output)
                heatmap_for_solver = heatmap_prob
                heatmap_for_viz = heatmap_prob
                edge_index_for_solve = knn_data.edge_index
                edge_index_for_viz = knn_data.edge_index

            utsp_tour = self.solver.model_solve(eval_data.coords, heatmap_for_solver, edge_index_for_solve)
            utsp_length = self.solver.calculate_tour_length(eval_data.coords, utsp_tour)
            
            nn_tour = self.solver.baseline_solve_nn(eval_data.coords)
            nn_length = self.solver.calculate_tour_length(eval_data.coords, nn_tour)
            
            elkai_tour, elkai_length = self.solver.baseline_solve_elkai(eval_data.coords)
            
            improvement = ((nn_length - utsp_length) / nn_length) * 100 if nn_length > 0 else 0
            
            metrics = {
                'loss': avg_loss,
                'utsp_tour_length': utsp_length,
                'nn_tour_length': nn_length,
                'elkai_tour_length': elkai_length,
                'improvement_vs_nn': improvement
            }
            
            if optimal_length:
                metrics['optimality_gap'] = ((utsp_length / optimal_length) - 1) * 100
            
            if not is_final:
                st.write(f"\n--- Epoch {epoch} Evaluation for {model_key} ---")
                st.write(f"Avg Training Loss: {avg_loss:.4f}")
                st.write(f"Your Model's Tour Length: {utsp_length:.4f}")
                st.write(f"Nearest Neighbor Baseline: {nn_length:.4f}")
                st.write(f"Elkai (LKH) Solver: {elkai_length:.4f}")
                st.write(f"Improvement vs. Baseline: {improvement:.2f}%")
                if optimal_length:
                    st.write(f"Optimality Gap: {metrics['optimality_gap']:.2f}%")
                st.write("-" * 30)
            
            metrics.update({
                'coords': eval_data.coords,
                'utsp_tour': utsp_tour,
                'nn_tour': nn_tour,
                'elkai_tour': elkai_tour,
                'heatmap': heatmap_for_solver,
                'edge_index': edge_index_for_solve
            })
            
            return metrics

def main():
    st.set_page_config(page_title="TSP Solver", layout="wide")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f"🖥️ Using device: {device}")
    
    st.title("🚀 Grand TSP Solver Configuration")
    
    # Add training status display
    if st.session_state.training_state['is_training']:
        st.info(f"""
        Training in progress:
        - Epoch: {st.session_state.training_state['current_epoch']}/{st.session_state.training_state['total_epochs']}
        - Batch: {st.session_state.training_state['current_batch']}/{st.session_state.training_state['total_batches']}
        - Current Loss: {st.session_state.training_state['current_loss']:.4f}
        - Best Tour Length: {st.session_state.training_state['best_tour_length']:.4f}
        """)
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["Single Training Run", "Experiment Suite"])
    
    with tab1:
        st.header("Single Training Run Configuration")
        
        # Data Source
        data_source = st.radio("Data Source:", ['Random (N=50)', 'berlin52'])
        
        # Model Configuration
        col1, col2 = st.columns(2)
        with col1:
            gnn_type = st.selectbox('GNN Model:', ['SAG','GAT','GIN','GraphSAGE'])
        with col2:
            loss_type = st.selectbox('Loss Function:', ['UTSPLoss','EdgeHeatmapLoss'])
        
        # Model Hyperparameters
        st.subheader("Model Hyperparameters")
        col1, col2 = st.columns(2)
        with col1:
            hidden_dim = st.slider('Hidden Dimension:', 32, 128, 64)
        with col2:
            n_layers = st.slider('Number of Layers:', 2, 5, 3)
        
        # Training Parameters
        st.subheader("Training Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            learning_rate = st.select_slider('Learning Rate:', 
                options=[1e-5, 1e-4, 1e-3], value=1e-4, format_func=lambda x: f"{x:.0e}")
        with col2:
            epochs = st.slider('Epochs:', 10, 300, 100, step=10)
        with col3:
            batch_size = st.slider('Batch Size:', 1, 128, 1)
        with col4:
            visualize_every = st.slider('Visualize Every:', 1, 50, 10)
        
        # Loss-specific parameters
        st.subheader("Loss-Specific Parameters")
        if loss_type == 'UTSPLoss':
            col1, col2 = st.columns(2)
            with col1:
                c1_penalty = st.slider('C1 Penalty:', 0.1, 5.0, 1.0)
            with col2:
                c2_penalty = st.slider('C2 Penalty:', 0.1, 5.0, 1.0)
            loss_params = {'c1_penalty': c1_penalty, 'c2_penalty': c2_penalty}
        else:
            degree_penalty = st.slider('Degree Penalty:', 0.5, 5.0, 2.0)
            loss_params = {'degree_penalty_weight': degree_penalty}
        
        # Dataset Parameters
        st.subheader("Dataset Parameters")
        num_samples = st.slider('Number of Samples:', 100, 2000, 500, step=50)
        
        # Training Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button('🚀 Start Training', type='primary', disabled=st.session_state.training_state['is_training']):
                with st.spinner('Initializing training...'):
                    optimal_length = None
                    if data_source == 'berlin52':
                        coords, optimal_length = TSPLIB_Parser().get_berlin52()
                        dataset = TSPDataset(coords=coords, num_samples=num_samples)
                    else:
                        dataset = TSPDataset(num_nodes=50, num_samples=num_samples)
                    
                    model = TSPModel(
                        gnn_type=gnn_type,
                        loss_type=loss_type,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers
                    )
                    
                    trainer = UTSPTrainer(model, device, visualizer=EnhancedVisualizer())
                    
                    if loss_type == 'UTSPLoss':
                        loss_params['temperature'] = 0.1
                    
                    final_metrics = trainer.train(
                        dataset,
                        loss_type=loss_type,
                        batch_size=batch_size,
                        epochs=epochs,
                        lr=learning_rate,
                        optimal_length=optimal_length,
                        visualize_every=visualize_every,
                        model_key=f"{gnn_type}+{loss_type}",
                        **loss_params
                    )
                    
                    if final_metrics:
                        st.success("Training completed!")
                        st.write("Final Results:")
                        st.write(f"Model Tour Length: {final_metrics['utsp_tour_length']:.4f}")
                        st.write(f"Nearest Neighbor Length: {final_metrics['nn_tour_length']:.4f}")
                        st.write(f"Elkai Solver Length: {final_metrics['elkai_tour_length']:.4f}")
                        st.write(f"Improvement vs. Baseline: {final_metrics['improvement_vs_nn']:.2f}%")
        
        with col2:
            if st.button('🛑 Stop Training', disabled=not st.session_state.training_state['is_training']):
                st.session_state.training_state['is_training'] = False
                st.rerun()
    
    with tab2:
        st.header("Experiment Suite Configuration")
        
        # Experiment Configuration
        st.subheader("Model Configurations")
        
        # Add new model configuration
        with st.expander("Add New Model Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                exp_gnn_type = st.selectbox('GNN Model:', ['SAG','GAT','GIN','GraphSAGE'], key='exp_gnn')
            with col2:
                exp_loss_type = st.selectbox('Loss Function:', ['UTSPLoss','EdgeHeatmapLoss'], key='exp_loss')
            
            col1, col2 = st.columns(2)
            with col1:
                exp_hidden_dim = st.slider('Hidden Dimension:', 32, 128, 64, key='exp_hidden')
            with col2:
                exp_n_layers = st.slider('Number of Layers:', 2, 5, 3, key='exp_layers')
            
            if st.button('Add Configuration'):
                config_key = f"{exp_gnn_type}+{exp_loss_type}"
                if config_key not in st.session_state.experiment_results:
                    st.session_state.experiment_results[config_key] = {
                        'model_params': {
                            'gnn_type': exp_gnn_type,
                            'loss_type': exp_loss_type,
                            'hidden_dim': exp_hidden_dim,
                            'n_layers': exp_n_layers
                        },
                        'train_params': {
                            'batch_size': 1,
                            'epochs': 100,
                            'lr': 1e-4,
                            'visualize_every': 10
                        }
                    }
                    st.success(f"Added configuration: {config_key}")
                else:
                    st.warning(f"Configuration {config_key} already exists")
        
        # Display and manage existing configurations
        if st.session_state.experiment_results:
            st.subheader("Current Configurations")
            for config_key, config in st.session_state.experiment_results.items():
                with st.expander(f"Configuration: {config_key}"):
                    st.write("Model Parameters:", config['model_params'])
                    st.write("Training Parameters:", config['train_params'])
                    if st.button('Remove Configuration', key=f"remove_{config_key}"):
                        del st.session_state.experiment_results[config_key]
                        st.rerun()
        
        # Run experiments
        if st.session_state.experiment_results:
            if st.button('Run Experiment Suite', type='primary', disabled=st.session_state.training_state['is_training']):
                with st.spinner('Running experiment suite...'):
                    # Prepare dataset
                    data_source = st.radio("Data Source:", ['Random (N=50)', 'berlin52'], key='exp_data')
                    num_samples = st.slider('Number of Samples:', 100, 2000, 500, step=50, key='exp_samples')
                    
                    optimal_length = None
                    if data_source == 'berlin52':
                        coords, optimal_length = TSPLIB_Parser().get_berlin52()
                        dataset = TSPDataset(coords=coords, num_samples=num_samples)
                    else:
                        dataset = TSPDataset(num_nodes=50, num_samples=num_samples)
                    
                    # Run each configuration
                    for config_key, config in st.session_state.experiment_results.items():
                        st.write(f"Running experiment: {config_key}")
                        model = TSPModel(**config['model_params'])
                        trainer = UTSPTrainer(model, device, visualizer=EnhancedVisualizer())
                        
                        final_metrics = trainer.train(
                            dataset,
                            model_key=config_key,
                            optimal_length=optimal_length,
                            **config['train_params']
                        )
                        
                        if final_metrics:
                            st.success(f"Completed experiment: {config_key}")
                            st.write(f"Final Tour Length: {final_metrics['utsp_tour_length']:.4f}")
                    
                    st.success("All experiments completed!")

if __name__ == "__main__":
    main() 