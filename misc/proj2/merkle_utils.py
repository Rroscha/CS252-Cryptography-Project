import hashlib

class MerkleProof:
    """The Merkle Proof to be provided to a verifier to 
       prove whether the leaf is at positions pos of the 
       Merkle Tree."""
    def __init__(self, leaf, pos, hashes):
        self.leaf = leaf     # data of leaf being checked
        self.pos  = pos      # the position in the tree of the leaf being checked
        self.hashes = hashes # the hashes from bottom to the top of tree for to
                             # prove the leaf is part of the Merkle Tree


##  The hash prefixes in the two functions above are a security measure.
##  They provide domain separation, meaning that the domain of a leaf hash
##  is seperated from the domain of an internal node hash.
##  This ensures that the verifier cannot mistake a leaf hash for 
##  an internal node hash, and vice versa. 

def hash_leaf(leaf):
    """hash a leaf value."""
    sha256 = hashlib.sha256()
    sha256.update(b"leaf:")   # hash prefix for a leaf
    sha256.update(leaf)
    return sha256.digest()


def hash_internal_node(left, right):
    """hash an internal node."""
    sha256 = hashlib.sha256()
    sha256.update(b"node:")   # hash prefix for an internal node
    sha256.update(left)
    sha256.update(right)
    return sha256.digest()

