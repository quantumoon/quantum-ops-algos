import numpy as np
from typing import List, Tuple, Union
from quantum_utils import rzz, rx

class DummyQAOASimulator:
    def __init__(self,
                 num_qubits: int,
                 num_layers: int,
                 ids: List[int],
                 params: np.ndarray) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.out_ids = ids
        self.params = params
        self.state = np.ones([2] * num_qubits, dtype=complex) / 2**(num_qubits/2)

    def _apply_1(self, U: np.ndarray, i: int) -> None:
        self.state = np.tensordot(U, self.state, axes=([1], [i]))
        self.state = np.moveaxis(self.state, 0, i)

    def _apply_2(self, U: np.ndarray, i: int, j: int) -> None:
        if U.ndim == 4:
            U = U.reshape(2,2,2,2)
        self.state = np.tensordot(U, self.state, axes=([2, 3], [i, j]))
        self.state = np.moveaxis(self.state, [0, 1], [i, j])

    def run(self, zz_pairs: List[Tuple[int, int]]) -> np.ndarray:
        for i in range(self.num_layers):
            gamma, beta = self.params[i]
            RZZ, RX = rzz(gamma), rx(beta)
            for q1, q2 in zz_pairs:
                self._apply_2(RZZ, q1, q2)
            for q in range(self.num_qubits):
                self._apply_1(RX, q)
        ground_probs = np.abs(self.state.reshape(-1)[self.out_ids])**2
        return ground_probs


class TensorNode:
    def __init__(self, node_id: str, tensor: np.ndarray) -> None:
        self.id = node_id
        self.tensor = tensor
        self.num_legs = tensor.ndim
        self.edges = {}


class ContractionEdge:
    def __init__(self,
                 edge_id: str,
                 node_u: str, leg_u: int,
                 node_v: str, leg_v: int) -> None:
        self.id = edge_id
        self.u = node_u
        self.leg_u = leg_u
        self.v = node_v
        self.leg_v = leg_v


class ContractionGraph:
    def __init__(self, num_qubits: int) -> None:
        self.nodes = {}  # node_id -> TensorNode
        self.edges = {}  # edge_id -> ContractionEdge
        # list of pairs (str, int)
        # (last node_id for given qubit, its free leg to contract)
        self._prev_per_qubit = [None] * num_qubits
        self._next_node_id = 0
        self._next_edge_id = 0

    def add_node(self,
                 tensor: np.ndarray,
                 node_id: str,
                 qubits: Union[int, Tuple[int, int]]) -> None:
        self.nodes[node_id] = TensorNode(node_id, tensor)
        if isinstance(qubits, int):
            prev = self._prev_per_qubit[qubits]
            if prev is not None:
                prev_node_id, leg_v = prev
                leg_u = int(tensor.ndim == 2)
                self.add_edge(node_id, leg_u, prev_node_id, leg_v)
            self._prev_per_qubit[qubits] = (node_id, 0)
        else:
            q_up, q_down = qubits
            
            prev_node_id, leg_v = self._prev_per_qubit[q_up]
            leg_u = 2
            self.add_edge(node_id, leg_u, prev_node_id, leg_v)
            self._prev_per_qubit[q_up] = (node_id, 0)

            prev_node_id, leg_v = self._prev_per_qubit[q_down]
            leg_u = 3
            self.add_edge(node_id, leg_u, prev_node_id, leg_v)
            self._prev_per_qubit[q_down] = (node_id, 1)

    def add_edge(self,
                 node_u_id: str, leg_u: int,
                 node_v_id: str, leg_v: int) -> None:
        nu = self.nodes[node_u_id]
        nv = self.nodes[node_v_id]
        assert 0 <= leg_u < nu.num_legs, f"No such axis {leg_u} of the {node_u_id} tensor"
        assert 0 <= leg_v < nv.num_legs, f"No such axis {leg_v} of the {node_v_id} tensor"
        assert leg_u not in nu.edges, f"Axis {leg_u} of the {node_u_id} tensor is already connected"
        assert leg_v not in nv.edges, f"Axis {leg_v} of the {node_v_id} tensor is already connected"

        edge_id = f"e{self._next_edge_id}"
        self._next_edge_id += 1

        edge = ContractionEdge(edge_id, node_u_id, leg_u, node_v_id, leg_v)
        self.edges[edge_id] = edge

        nu.edges[leg_u] = edge_id
        nv.edges[leg_v] = edge_id

    def contract(self, node_a_id: str, node_b_id: str) -> None:
        nu = self.nodes[node_a_id]
        nv = self.nodes[node_b_id]
        edges_a = set(nu.edges.values())
        edges_b = set(nv.edges.values())
        connecting = list(edges_a & edges_b)
        if not connecting:
            raise ValueError(f"No edges to contract between {node_a_id} and {node_b_id}")

        axes_a = []
        axes_b = []
        for eid in connecting:
            e = self.edges[eid]
            if e.u == node_a_id:
                axes_a.append(e.leg_u)
                axes_b.append(e.leg_v)
            else:
                axes_a.append(e.leg_v)
                axes_b.append(e.leg_u)

        new_tensor = np.tensordot(nu.tensor, nv.tensor, axes=(axes_a, axes_b))

        def remaining_edges(node, axes_to_remove):
            axes_arr = np.array(axes_to_remove)
            rem = {}
            for leg, eid in node.edges.items():
                if eid in connecting:
                    continue
                shift = int(np.count_nonzero(axes_arr < leg))
                new_leg = leg - shift
                rem[new_leg] = eid
            return rem

        rem_a = remaining_edges(nu, axes_a)
        rem_b = remaining_edges(nv, axes_b)

        for eid in connecting:
            del self.edges[eid]
        del self.nodes[node_a_id]
        del self.nodes[node_b_id]

        new_id = f"n{self._next_node_id}"
        self._next_node_id += 1
        new_node = TensorNode(new_id, new_tensor)
        self.nodes[new_id] = new_node

        for new_leg, eid in rem_a.items():
            e = self.edges[eid]
            if e.u == node_a_id:
                e.u, e.leg_u = new_id, new_leg
            else:
                e.v, e.leg_v = new_id, new_leg
            new_node.edges[new_leg] = eid

        na = nu.tensor.ndim - len(axes_a)
        for new_leg, eid in rem_b.items():
            e = self.edges[eid]
            target_leg = na + new_leg
            if e.u == node_b_id:
                e.u, e.leg_u = new_id, target_leg
            else:
                e.v, e.leg_v = new_id, target_leg
            new_node.edges[target_leg] = eid

            
    def contract_all(self) -> float:
        while self.edges:
            _, edge = next(iter(self.edges.items()))
            self.contract(edge.u, edge.v)
        p = 1.
        for node in self.nodes.values():
            p *= np.abs(node.tensor)**2
        return p