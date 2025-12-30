$$
\[ |\det(i(J_n - 2I_n))| = |E(Q_n) - V(Q_n)| = |n-2| \cdot 2^{n-1} \]
$$

### **Theorem: The Imaginary Vybn Matrix**

**Definition 1 (The Operator):**
Let \(A_n\) be the \(n \times n\) complex matrix defined by:

$$
\[ A_n = i(J_n - 2I_n) \]
$$

where \(J_n\) is the all-ones matrix and \(I_n\) is the identity matrix.

**Definition 2 (The Geometry):**
Let \(Q_n\) be the Boolean Hypercube of dimension \(n\).
*   Vertices: \(V_n = 2^n\)
*   Edges: \(E_n = n \cdot 2^{n-1}\)

**The Correspondence:**
For all \(n \ge 1\), the magnitude of the determinant of \(A_n\) is exactly equal to the absolute difference between the edges and vertices of \(Q_n\).

$$
\[ |\det(A_n)| = |E_n - V_n| \]
$$

**The Structural Form:**
This quantity reduces to a sequential integer scaling law.
Let \(k = n-2\). The magnitude is given by:

\[ |\det(A_n)| = k \cdot 2^{k+1} \]

**Significance:**
This establishes an exact isomorphism between the spectral volume of the rank-shifted matrix \(i(J-2I)\) and the topological density of the \(n\)-dimensional computational basis. The sequence \(k \cdot 2^{k+1}\) represents the fundamental counting unit of hypercube connectivity relative to its capacity.
