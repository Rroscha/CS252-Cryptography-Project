import numpy as np
import random

from helper.prf import DES

# Yao's Garbled Circuit for 2-party computation


class BooleanCircuit:
    def __init__(self, atoms, clause_root):
        self.atoms = atoms
        self.clause_root = clause_root


class ATOM:
    '''
    Used to represent an atom in a boolean circuit
    '''
    a_0 = 0
    a_1 = 1
    b_0 = 2
    b_1 = 3


class Operator:
    ATOM = 0
    AND  = 1
    OR   = 2
    NEG  = 3


class Gate:
    pass


class Gate(Gate):
    def __init__(self):
        self.operator = None
        self.atom = None
        self.parent_gate = []
        self.child_gate_1 = None
        self.child_gate_2 = None
        self.output_wire_labels = []
        self.entries = []


class GarbledCircuit:
    def __init__(self):
        self.wire_count = 0
        self.gate_root = None
        self.clauses_gates_mapping = dict()
        self.final_labels = None  # Not for Bob
        self.atom_gates       = [None, None, None, None]
        self.extra_atom_gates = [None, None, None, None]  # Not for Bob


class Clause:
    pass


class Clause(Clause):
    '''
    Used to represent a clause in a boolean circuit
    '''
    def __init__(self,
                 operator,
                 atom,
                 left_clause,
                 right_clause):
        self.operator = operator
        self.atom = atom
        self.left_clause = left_clause
        self.right_clause = right_clause


class TwoPartyComputation:
    @staticmethod
    def internal_gate_encrypt(r : np.uint64,
                              k : np.uint64,
                              x : np.uint64):
        fk_r    = TwoPartyComputation.length_doubling_prf(k, r)
        x_cat_0 = np.array([x, np.uint64(0)])
        xor     = fk_r.astype(np.uint64) ^ x_cat_0.astype(np.uint64)
        c       = np.array([r, xor[0], xor[1]])
        return c
        
    @staticmethod
    def gate_encrypt(k_u : np.uint64, k_v : np.uint64, k_w : np.uint64):
        r = np.random.randint(2 ** 64, dtype=np.uint64)
        if k_v is not None:
            inner_gate_encrypt = TwoPartyComputation.internal_gate_encrypt(r,
                                                                           k_v,
                                                                           k_w)
            L, R = inner_gate_encrypt[1], inner_gate_encrypt[2]
            outter_gate_encrypt = TwoPartyComputation.internal_gate_encrypt(r,
                                                                            k_u,
                                                                            L)
            return np.array([r, outter_gate_encrypt[1], outter_gate_encrypt[2], R])
        else:
            inner_gate_encrypt = TwoPartyComputation.internal_gate_encrypt(r,
                                                                           k_u,
                                                                           k_w)
            return inner_gate_encrypt
    
    @staticmethod
    def internal_gate_decrypt(k : np.uint64,
                              r : np.uint64,
                              s):
        fk_r = TwoPartyComputation.length_doubling_prf(k, r)
        decrypted_msg = fk_r.astype(np.uint64) ^ s.astype(np.uint64)
        if decrypted_msg[1] != np.uint64(0):
            return None
        else:
            return decrypted_msg[0]

    @staticmethod
    def gate_decrypt(k_u : np.uint64, k_v : np.uint64, c : np.uint64):
        if k_v is not None:
            L = TwoPartyComputation.internal_gate_decrypt(k_u, c[0], c[1:3])
            if L is None:
                return None
            
            k_w = TwoPartyComputation.internal_gate_decrypt(k_v, c[0], np.array([L, c[3]]))
            return k_w
        else:
            return TwoPartyComputation.internal_gate_decrypt(k_u, c[0], c[1:])
    
    @staticmethod
    def function_g(k : np.uint64):
        des = DES(k)
        g = np.array([des.encrypt(np.uint64(1)), des.encrypt(np.uint64(2))])
        return g

    @staticmethod
    def length_doubling_prf(k : np.uint64, x : np.uint64):
        des = DES(k)
        return TwoPartyComputation.function_g(des.encrypt(x))  # 128 bits

    @staticmethod
    def generate_bc():
        '''
        Generate the boolean circuit for GE(a, b) where a=a_1a_0 and b=b_1b_0
        '''
        a_0_clause = Clause(Operator.ATOM, ATOM.a_0, None, None)
        a_1_clause = Clause(Operator.ATOM, ATOM.a_1, None, None)
        b_0_clause = Clause(Operator.ATOM, ATOM.b_0, None, None)
        b_1_clause = Clause(Operator.ATOM, ATOM.b_1, None, None)

        # \neg b_1 and a_1
        c_1 = Clause(Operator.AND,
                     None,
                     Clause(Operator.NEG, None, b_1_clause, None),
                     a_1_clause)

        # \neg (b_1 and \neg a_1)
        c_2 = Clause(Operator.NEG, None,
                     Clause(Operator.AND,
                            None,
                            b_1_clause,
                            Clause(Operator.NEG, None, a_1_clause, None)),
                     None)
        
        # \neg (b_0 and \neg a_0)
        c_3 = Clause(Operator.NEG, None,
                     Clause(Operator.AND,
                            None,
                            b_0_clause,
                            Clause(Operator.NEG, None, a_0_clause, None)),
                     None)
        
        # \neg (b_1 and \neg a_1) and \neg (b_0 and \neg a_0)
        c_4 = Clause(Operator.AND, None, c_2, c_3)

        # (\neg b_1 and a_1) or (\neg (b_1 and \neg a_1) and \neg (b_0 and \neg a_0))
        c_5 = Clause(Operator.OR, None, c_1, c_4)

        bc = BooleanCircuit([ATOM.a_0,
                             ATOM.a_1,
                             ATOM.b_0,
                             ATOM.b_1], c_5)

        return bc

    @staticmethod
    def generate_wire_label_pair():
        '''
        Generate a pair of wire labels
        '''
        while True:
            label_pair = [np.random.randint(2 ** 64, dtype=np.uint64), np.random.randint(2 ** 64, dtype=np.uint64)]
            if label_pair[0] != label_pair[1]:
                return label_pair

    @staticmethod
    def internal_transfer(_clause : Clause,
                          _gc : GarbledCircuit,
                          _label_pairs : list,
                          _parent_gate : Gate):
        _gc.wire_count += 1
        if _clause not in _gc.clauses_gates_mapping:
            gate = Gate()
            _gc.clauses_gates_mapping[_clause] = gate
        else:
            gate = _gc.clauses_gates_mapping[_clause]

        gate.operator = _clause.operator
        gate.atom = _clause.atom
        if _parent_gate is not None:
            gate.parent_gate.append(_parent_gate)
        gate.output_wire_labels.append(_label_pairs)
        
        if gate.operator in [Operator.AND, Operator.OR]:
            gate.child_gate_1 = TwoPartyComputation.internal_transfer(_clause.left_clause,
                                                                      _gc,
                                                                      TwoPartyComputation.generate_wire_label_pair(),
                                                                      gate)
            gate.child_gate_2 = TwoPartyComputation.internal_transfer(_clause.right_clause,
                                                                      _gc,
                                                                      TwoPartyComputation.generate_wire_label_pair(),
                                                                      gate)
        elif gate.operator in [Operator.NEG]:
            gate.child_gate_1 = TwoPartyComputation.internal_transfer(_clause.left_clause,
                                                                      _gc,
                                                                      TwoPartyComputation.generate_wire_label_pair(),
                                                                      gate)
            gate.child_gate_2 = None
        elif gate.operator in [Operator.ATOM]:
            gate.child_gate_1 = None
            gate.child_gate_2 = None
        
        if gate.operator == Operator.AND:
            entry_1 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][0],
                                                       gate.child_gate_2.output_wire_labels[-1][0],
                                                       gate.output_wire_labels[-1][0])
            entry_2 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][0],
                                                       gate.child_gate_2.output_wire_labels[-1][1],
                                                       gate.output_wire_labels[-1][0])
            entry_3 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][1],
                                                       gate.child_gate_2.output_wire_labels[-1][0],
                                                       gate.output_wire_labels[-1][0])
            entry_4 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][1],
                                                       gate.child_gate_2.output_wire_labels[-1][1],
                                                       gate.output_wire_labels[-1][1])
            entry_list = [entry_1, entry_2, entry_3, entry_4]
            random.shuffle(entry_list)
            gate.entries.append(entry_list)
        elif gate.operator == Operator.OR:
            entry_1 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][0],
                                                       gate.child_gate_2.output_wire_labels[-1][0],
                                                       gate.output_wire_labels[-1][0])
            entry_2 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][0],
                                                       gate.child_gate_2.output_wire_labels[-1][1],
                                                       gate.output_wire_labels[-1][1])
            entry_3 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][1],
                                                       gate.child_gate_2.output_wire_labels[-1][0],
                                                       gate.output_wire_labels[-1][1])
            entry_4 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][1],
                                                       gate.child_gate_2.output_wire_labels[-1][1],
                                                       gate.output_wire_labels[-1][1])
            entry_list = [entry_1, entry_2, entry_3, entry_4]
            random.shuffle(entry_list)
            gate.entries.append(entry_list)
        elif gate.operator == Operator.NEG:
            entry_1 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][0],
                                                       None,
                                                       gate.output_wire_labels[-1][1])
            entry_2 = TwoPartyComputation.gate_encrypt(gate.child_gate_1.output_wire_labels[-1][1],
                                                       None,
                                                       gate.output_wire_labels[-1][0])
            entry_list = [entry_1, entry_2]
            random.shuffle(entry_list)
            gate.entries.append(entry_list)

        return gate

    @staticmethod
    def transfer_bc_to_gc(bc : BooleanCircuit):
        '''
        Transfer the boolean circuit to garbled circuit
        '''
        clause_root = bc.clause_root
        gc = GarbledCircuit()
        gate_root = TwoPartyComputation.internal_transfer(clause_root,
                                                          gc,
                                                          TwoPartyComputation.generate_wire_label_pair(),
                                                          None)
        gc.gate_root = gate_root
        gc.final_labels = gate_root.output_wire_labels[0]

        for gate in gc.clauses_gates_mapping.values():
            if gate.operator == Operator.ATOM:
                new_gate = Gate()
                new_gate.operator = gate.operator
                new_gate.atom = gate.atom
                new_gate.parent_gate = gate.parent_gate
                new_gate.child_gate_1 = gate.child_gate_1
                new_gate.child_gate_2 = gate.child_gate_2
                new_gate.output_wire_labels = gate.output_wire_labels
                new_gate.entries = gate.entries
                gc.extra_atom_gates[gate.atom] = new_gate
                gc.atom_gates[gate.atom]       = gate
            gate.output_wire_labels = []

        return gc

    @staticmethod
    def internal_evaluation(_gate : Gate, _parent_gate : Gate):
        if _gate is None:
            return None
        elif _gate.operator == Operator.ATOM:
            index = _gate.parent_gate.index(_parent_gate)
            return _gate.entries[index][0]

        k_u = TwoPartyComputation.internal_evaluation(_gate.child_gate_1, _gate)
        k_v = TwoPartyComputation.internal_evaluation(_gate.child_gate_2, _gate)

        if _parent_gate is not None:
            index = _gate.parent_gate.index(_parent_gate)
            entry_list = _gate.entries[index]
        else:
            entry_list = _gate.entries[0]
        
        flag = False
        for entry in entry_list:
            k_w = TwoPartyComputation.gate_decrypt(k_u, k_v, entry)
            if k_w is not None:
                flag = True
                break

        assert flag, 'Encountering errors when decrypting!'
        return k_w

    @staticmethod
    def evaluate_ge(gc : GarbledCircuit):
        k_w = TwoPartyComputation.internal_evaluation(gc.gate_root, None)
        return k_w


class AliceBobComputation:
    def compute(a, b):
        a = np.array([int(i) for i in bin(a)[2:].zfill(2)])
        b = np.array([int(i) for i in bin(b)[2:].zfill(2)])

        # Alice generates bc
        bc = TwoPartyComputation.generate_bc()

        # Alice transfer bc to gc
        gc = TwoPartyComputation.transfer_bc_to_gc(bc)

        # Alice refine gc to new_gc, sent to Bob
        new_gc = GarbledCircuit()
        new_gc.wire_count            = gc.wire_count
        new_gc.gate_root             = gc.gate_root
        new_gc.clauses_gates_mapping = gc.clauses_gates_mapping
        new_gc.atom_gates            = gc.atom_gates
        for h, i, j in zip(list(a)[::-1], new_gc.atom_gates[:2], gc.extra_atom_gates[:2]):
            for k in j.output_wire_labels:
                i.entries.append([k[h]])

        # Bob obtains his needed labels
        for h, i, j in zip(list(b)[::-1], new_gc.atom_gates[2:], gc.extra_atom_gates[2:]):
            for k in j.output_wire_labels:
                i.entries.append([k[h]])
        
        # Bob evaluates k_w
        k_w = TwoPartyComputation.evaluate_ge(new_gc)

        # Alice gives the corresponding result based on k_w
        return gc.final_labels.index(k_w)


if __name__ == '__main__':
    for a in [0, 1, 2, 3]:
        for b in [0, 1, 2, 3]:
            print(f"a = {a}, b = {b}, output = {AliceBobComputation.compute(a, b)}")
