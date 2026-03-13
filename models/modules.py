"""Neural backbone blocks for spatiotemporal battery representation learning."""

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


# -----------------------------------------------------------------------------
# Base class for battery dynamical backbones
# -----------------------------------------------------------------------------
class BSTBackbone(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for lithium-ion battery dynamical evolution modules.

    Input:
        x   : (B, L, N, D_in)
        adj : (N, N) or None

    Output:
        h   : (B, L, N, D_hidden)
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        require_adj: bool = False,
        name: str = "battery_dynamics",
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.require_adj = bool(require_adj)
        self.name = str(name)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        """
        Public forward with basic input / output validation.

        Args:
            x   : (B, L, N, D_in)
            adj : (N, N) or None

        Returns:
            h   : (B, L, N, D_hidden)
        """
        if x.dim() != 4:
            raise ValueError(
                f"[{self.name}] `x` must be 4D (B, L, N, D_in), got {tuple(x.shape)}"
            )
        B, L, N, D_in = x.shape
        if D_in != self.d_in:
            raise ValueError(
                f"[{self.name}] D_in mismatch: expected {self.d_in}, got {D_in}"
            )

        if self.require_adj and adj is None:
            raise ValueError(f"[{self.name}] `adj` is required but got None.")

        if adj is not None:
            if adj.dim() != 2:
                raise ValueError(
                    f"[{self.name}] `adj` must be 2D (N, N), got {tuple(adj.shape)}"
                )
            if adj.shape[0] != N or adj.shape[1] != N:
                raise ValueError(
                    f"[{self.name}] `adj` shape mismatch: expected ({N}, {N}), "
                    f"got {tuple(adj.shape)}"
                )
            adj = adj.to(device=x.device, dtype=x.dtype)

        h = self._forward_impl(x, adj)

        if h.dim() != 4:
            raise ValueError(
                f"[{self.name}] output must be 4D (B, L, N, D_hidden), got {tuple(h.shape)}"
            )
        B2, L2, N2, D_h = h.shape
        if (B2, L2, N2) != (B, L, N):
            raise ValueError(
                f"[{self.name}] output (B, L, N) mismatch: "
                f"input (B,L,N)=({B},{L},{N}), output=({B2},{L2},{N2})"
            )
        if D_h != self.d_hidden:
            raise ValueError(
                f"[{self.name}] D_hidden mismatch: expected {self.d_hidden}, got {D_h}"
            )

        return h

    @abc.abstractmethod
    def _forward_impl(self, x: Tensor, adj: Optional[Tensor]) -> Tensor:
        """
        Core computation.

        Args:
            x   : (B, L, N, D_in)
            adj : (N, N) or None

        Returns:
            h   : (B, L, N, D_hidden)
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Node-wise self-attention block (Transformer-style)
# -----------------------------------------------------------------------------
class NodeSelfAttentionBlock(nn.Module):
    """
    Self-attention over the node dimension N (for each (B, L) independently),
    followed by a position-wise feed-forward network.

    Input:
        x        : (B_flat, N, D_model), where B_flat = B * L
        attn_bias: optional additive attention bias, broadcastable to
                   (B_flat, n_heads, N, N). Typically shape (1, 1, N, N).

    Output:
        out          : (B_flat, N, D_model)
        attn_weights : (B_flat, n_heads, N, N) if need_weights=True, else None
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x         : (B_flat, N, D_model)
            attn_bias : additive bias for attention logits, can be None or
                        broadcastable to (B_flat, n_heads, N, N)
            need_weights: whether to return attention weights

        Returns:
            out          : (B_flat, N, D_model)
            attn_weights : (B_flat, n_heads, N, N) or None
        """
        B_flat, N, D = x.shape

        # ----- Multi-head self-attention -----
        residual = x
        x_norm = self.norm1(x)  # (B_flat, N, D)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape to (B_flat, n_heads, N, d_head)
        q = q.view(B_flat, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B_flat, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B_flat, N, self.n_heads, self.d_head).transpose(1, 2)

        # Attention logits: (B_flat, n_heads, N, N)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)

        # Additive attention bias (e.g., alpha*A + beta*DeltaA)
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # (B_flat, n_heads, N, d_head)
        context = context.transpose(1, 2).contiguous().view(B_flat, N, D)

        out = self.out_proj(context)
        out = residual + out  # first residual connection

        # ----- Feed-forward network -----
        residual2 = out
        out_norm = self.norm2(out)
        out_ffn = self.ffn(out_norm)
        out = residual2 + out_ffn  # second residual connection

        if need_weights:
            return out, attn_weights
        else:
            return out, None


# -----------------------------------------------------------------------------
# BiLSTM over time + node-wise Transformer with adj prior + learnable residual
# -----------------------------------------------------------------------------
class NodeAttnLSTM(BSTBackbone):
    """
    Battery dynamics backbone:

    1) Per-node bidirectional LSTM over the time dimension (shared parameters
       across nodes) to model temporal dynamics for each cell.
    2) Node-wise Transformer over the node dimension at each time step, with
       adjacency-based attention bias:

           logits_ij += alpha * A_ij + beta * DeltaA_ij

       where A is the (possibly binarized) adjacency with self-loops, and
       DeltaA is a learnable residual matrix.

    Additionally, node-wise positional embeddings are based on the physical
    adjacency (via node degrees) and learnable base embeddings.

    Input:
        x   : (B, L, N, D_in)
        adj : (N, N) or None

    Output:
        h   : (B, L, N, D_hidden)

    Extra:
        forward_with_attn(x, adj) -> (h, attn)
        where attn: (B, L, n_heads, N, N) for the last Transformer layer.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_nodes: int,
        n_lstm_layers: int = 1,
        bidirectional: bool = True,
        lstm_dropout: float = 0.0,
        n_heads: int = 4,
        n_trans_layers: int = 2,
        dim_feedforward: int = 256,
        trans_dropout: float = 0.1,
        require_adj: bool = False,
        use_adj_bias: bool = True,
        alpha_init: float = 1.0,
        beta_init: float = 0.1,
        name: str = "lstm_node_transformer",
    ) -> None:
        """
        Args:
            d_in            : input feature dimension per node
            d_hidden        : final hidden dimension per node
            n_nodes         : number of nodes (cells), fixed for this backbone
            n_lstm_layers   : number of LSTM layers
            bidirectional   : whether to use bidirectional LSTM
            lstm_dropout    : LSTM dropout between layers
            n_heads         : number of attention heads in node-wise Transformer
            n_trans_layers  : number of stacked Transformer blocks
            dim_feedforward : hidden dimension of Transformer FFN
            trans_dropout   : dropout inside Transformer blocks
            require_adj     : whether `adj` is required in forward()
            use_adj_bias    : whether to use adjacency-based attention bias
            alpha_init      : initial value for alpha in (alpha * A)
            beta_init       : initial value for beta in (beta * DeltaA)
        """
        super().__init__(
            d_in=d_in,
            d_hidden=d_hidden,
            require_adj=require_adj,  # adj is optional; if None, no prior bias is used
            name=name,
        )

        self.n_nodes = int(n_nodes)
        self.use_adj_bias = bool(use_adj_bias)
        self.n_heads = int(n_heads)

        # ----- 1) LSTM over time (per node) -----
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_hidden // 2 if bidirectional else d_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if n_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,  # bi-directional LSTM
        )

        # ----- Node positional embeddings (based on adjacency degrees) -----
        # Base learnable embedding for each node, shape: (N, D_hidden)
        self.node_emb_base = nn.Parameter(torch.randn(self.n_nodes, d_hidden) * 0.01)

        # ----- 2) Node-wise Transformer over nodes -----
        self.trans_layers = nn.ModuleList(
            [
                NodeSelfAttentionBlock(
                    d_model=d_hidden,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=trans_dropout if n_trans_layers > 1 else 0.0,
                )
                for _ in range(n_trans_layers)
            ]
        )

        # ----- Adjacency prior + learnable residual for attention logits -----
        # Learnable residual adjacency: DeltaA (N, N)
        self.delta_A = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))

        # Learnable scalars alpha and beta:
        # logits += alpha * A + beta * DeltaA
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # Public API: base forward (BatteryDynamicsBackbone)
    # ------------------------------------------------------------------ #
    def _forward_impl(self, x: Tensor, adj: Optional[Tensor]) -> Tensor:
        """Run the shared backbone without returning attention weights."""
        h, _ = self._core_forward(x, adj, need_attn=False)
        return h

    # ------------------------------------------------------------------ #
    # Public API: forward with attention weights
    # ------------------------------------------------------------------ #
    def forward_with_attn(
        self, x: Tensor, adj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that additionally returns the node-wise attention weights
        from the last Transformer layer.

        Args:
            x   : (B, L, N, D_in)
            adj : (N, N) or None

        Returns:
            h    : (B, L, N, D_hidden)
            attn : (B, L, n_heads, N, N)  # attention weights of the last layer
        """
        # Basic consistency checks (similar to base class forward)
        if x.dim() != 4:
            raise ValueError(
                f"[{self.name}] `x` must be 4D (B, L, N, D_in), got {tuple(x.shape)}"
            )
        B, L, N, D_in = x.shape
        if D_in != self.d_in:
            raise ValueError(
                f"[{self.name}] D_in mismatch: expected {self.d_in}, got {D_in}"
            )
        if N != self.n_nodes:
            raise ValueError(
                f"[{self.name}] n_nodes mismatch: backbone was created with "
                f"n_nodes={self.n_nodes}, but got N={N} in input."
            )

        if adj is not None:
            if adj.dim() != 2:
                raise ValueError(
                    f"[{self.name}] `adj` must be 2D (N, N), got {tuple(adj.shape)}"
                )
            if adj.shape[0] != N or adj.shape[1] != N:
                raise ValueError(
                    f"[{self.name}] `adj` shape mismatch: expected ({N}, {N}), "
                    f"got {tuple(adj.shape)}"
                )
            adj = adj.to(device=x.device, dtype=x.dtype)

        h, attn = self._core_forward(x, adj, need_attn=True)

        # attn is guaranteed not None if need_attn=True
        assert attn is not None
        return h, attn

    # ------------------------------------------------------------------ #
    # Core computation: BiLSTM over time + node-wise Transformer
    # ------------------------------------------------------------------ #
    def _core_forward(
        self,
        x: Tensor,
        adj: Optional[Tensor],
        need_attn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Core logic:

        1) Apply per-node BiLSTM over the time dimension.
        2) Add node positional embeddings (based on adjacency degrees).
        3) Apply node-wise Transformer with adjacency prior + learnable
           residual as attention bias.

        Args:
            x        : (B, L, N, D_in)
            adj      : (N, N) or None
            need_attn: whether to return attention weights from the last layer

        Returns:
            h       : (B, L, N, D_hidden)
            attn    : (B, L, n_heads, N, N) or None
        """
        B, L, N, D_in = x.shape
        device = x.device
        dtype = x.dtype

        if N != self.n_nodes:
            raise ValueError(
                f"[{self.name}] n_nodes mismatch: backbone was created with "
                f"n_nodes={self.n_nodes}, but got N={N} in input."
            )

        # ---------- Step 1: per-node BiLSTM over time ---------- #
        # Treat each node as an independent sequence:
        # x: (B, L, N, D_in) -> (B*N, L, D_in)
        x_node = rearrange(x, "b l n d -> (b n) l d")

        lstm_out, _ = self.lstm(x_node)  # (B*N, L, 2*lstm_hidden)

        # Back to (B, L, N, d_hidden)
        h_time = rearrange(lstm_out, "(b n) l d -> b l n d", b=B, n=N)
        # h_time: (B, L, N, d_hidden)

        # ---------- Node positional embedding based on adjacency ---------- #
        # node_emb_base: (N, d_hidden)
        if adj is not None:
            eye = torch.eye(N, device=device, dtype=dtype)
            adj_with_self = (adj + eye).clamp(max=1.0)  # (N, N)

            # Degree per node: (N, 1), then normalized
            deg = adj_with_self.sum(dim=-1, keepdim=True)  # (N, 1)
            deg_mean = deg.mean().clamp(min=1e-6)
            deg_norm = deg / deg_mean  # (N, 1)

            # node_pos: (N, d_hidden)
            node_pos = deg_norm * self.node_emb_base
        else:
            node_pos = self.node_emb_base  # (N, d_hidden)

        # Broadcast node_pos to (B, L, N, d_hidden)
        node_pos_broadcast = rearrange(node_pos, "n d -> 1 1 n d")
        h_time = h_time + node_pos_broadcast

        # ---------- Step 2: node-wise Transformer with adj prior ---------- #
        # Flatten (B, L) into a single batch dimension:
        # (B, L, N, d_hidden) -> (B*L, N, d_hidden)
        h_nodes = rearrange(h_time, "b l n d -> (b l) n d")

        # Prepare adjacency-based attention bias if requested
        if self.use_adj_bias and adj is not None:
            eye = torch.eye(N, device=device, dtype=dtype)
            adj_with_self = (adj + eye).clamp(max=1.0)  # (N, N)

            # Symmetrize learnable residual DeltaA
            delta_A = 0.5 * (self.delta_A + self.delta_A.t())  # (N, N)

            # logits bias: alpha * A + beta * DeltaA
            attn_bias_matrix = self.alpha * adj_with_self.to(
                torch.float32
            ) + self.beta * delta_A.to(torch.float32)
            attn_bias_matrix = attn_bias_matrix.to(device=device, dtype=dtype)

            # (1, 1, N, N) so it can broadcast to (B_flat, n_heads, N, N)
            attn_bias = rearrange(attn_bias_matrix, "i j -> 1 1 i j")
        else:
            attn_bias = None

        last_attn: Optional[Tensor] = None

        # Stacked Transformer layers over nodes
        for i, layer in enumerate(self.trans_layers):
            is_last = i == len(self.trans_layers) - 1
            layer_need_weights = need_attn and is_last

            h_nodes, attn_weights = layer(
                h_nodes,
                attn_bias=attn_bias,
                need_weights=layer_need_weights,
            )
            if layer_need_weights:
                last_attn = attn_weights  # (B_flat, n_heads, N, N)

        # Back to (B, L, N, d_hidden)
        h = rearrange(h_nodes, "(b l) n d -> b l n d", b=B, l=L)

        # Arrange attention to (B, L, n_heads, N, N) if needed
        if need_attn and last_attn is not None:
            attn = rearrange(last_attn, "(b l) h n1 n2 -> b l h n1 n2", b=B, l=L)
        else:
            attn = None

        return h, attn


if __name__ == "__main__":

    # Example configuration
    B, L, N, D_in = 1024, 200, 11, 3
    D_hidden = 32
    lstm_hidden = 32

    backbone = NodeAttnLSTM(
        d_in=D_in,
        d_hidden=D_hidden,
        n_nodes=N,
        n_lstm_layers=1,
        bidirectional=True,
        lstm_dropout=0.0,
        n_heads=4,
        n_trans_layers=1,
        dim_feedforward=128,
        trans_dropout=0.1,
        require_adj=True,
        use_adj_bias=True,
        alpha_init=1.0,
        beta_init=0.1,
    )

    x = torch.randn(B, L, N, D_in)
    adj = torch.ones(N, N)  # or your real battery topology

    # Only features:
    h = backbone(x, adj)  # (B, L, N, D_hidden)
    print("h", h.shape)

    # Features + attention weights:
    h, attn = backbone.forward_with_attn(x, adj)
    print("h", h.shape)
    print("attn", attn.shape)
    # h   : (B, L, N, D_hidden)
    # attn: (B, L, n_heads, N, N)
