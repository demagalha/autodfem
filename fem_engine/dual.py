import numpy as np

class DualTensor:
    """
    Vectorized Dual Number class wrapping NumPy arrays.
    real: np.ndarray of shape (d1, d2, ...)
    dual: np.ndarray of shape (d1, d2, ..., N_vars)
    """
    __array_priority__ = 10000
    __array_ufunc__ = None

    def __init__(self, real, dual):
        self.real = np.asarray(real, dtype=float)
        self.dual = np.asarray(dual, dtype=float)

    def __repr__(self):
        return f"DualTensor(shape={self.real.shape}, dual_shape={self.dual.shape})"

    def __array_function__(self, func, types, args, kwargs):
        """Intercepts standard NumPy functions and redirects them."""
        # np.outer
        if func is np.outer:
            a, b = args
            
            # Format 'a' as a column vector
            if isinstance(a, DualTensor):
                a_col = a.flatten()[:, None]
            else:
                a_col = np.asarray(a).flatten()[:, None]
                
            # Format 'b' as a row vector
            if isinstance(b, DualTensor):
                b_row = b.flatten()[None, :]
            else:
                b_row = np.asarray(b).flatten()[None, :]
                
            # Multiply using DualTensor's safe __mul__ method
            return a_col * b_row
            
        # np.trace
        if func is np.trace:
            a = args[0]
            new_real = np.trace(a.real, **kwargs)
            new_dual = np.trace(a.dual, axis1=0, axis2=1)
            return DualTensor(new_real, new_dual)

        # np.dot
        if func is np.dot:
            a, b = args
            
            # Determine dimensions safely whether they are DualTensors or raw NumPy arrays
            a_ndim = np.ndim(getattr(a, 'real', a))
            b_ndim = np.ndim(getattr(b, 'real', b))
            
            if a_ndim == 1 and b_ndim == 1:
                # 1D Vector dot product: element-wise multiply (using safe __mul__), then sum
                ab = a * b
                new_real = np.sum(ab.real)
                new_dual = np.sum(ab.dual, axis=0) # Sum across the spatial dimension, keep AD dimension
                return DualTensor(new_real, new_dual)
            else:
                # For 2D tensors, np.dot is mathematically identical to matrix multiplication (@)
                return a @ b
            
        return NotImplemented

    # Some properties needed for weak forms
    @property
    def T(self):
        """Transposes the real part and the corresponding spatial dimensions of the dual part."""
        new_real = self.real.T
        new_dual = np.transpose(self.dual, axes=(1, 0, 2))
        return DualTensor(new_real, new_dual)
        
    def __getitem__(self, key):
        return DualTensor(self.real[key], self.dual[key])

    def __setitem__(self, key, value):
        if isinstance(value, DualTensor):
            self.real[key] = value.real
            self.dual[key] = value.dual
        else:
            self.real[key] = value
            # If assigning a pure float, the derivative is zero
            self.dual[key] = 0.0

    def reshape(self, shape):
        new_real = self.real.reshape(shape)
        # The dual dimension (the last one) must remain unchanged
        new_dual = self.dual.reshape(shape + (self.dual.shape[-1],))
        return DualTensor(new_real, new_dual)

    def flatten(self):
        new_real = self.real.flatten()
        new_dual = self.dual.reshape((new_real.size, self.dual.shape[-1]))
        return DualTensor(new_real, new_dual)

    def det(self):
        """Determinant of a NxN DualTensor matrix."""
        if self.real.ndim != 2 or self.real.shape[0] != self.real.shape[1]:
            raise ValueError("det() requires a square matrix.")
            
        det_real = np.linalg.det(self.real)
        inv_real = np.linalg.inv(self.real)
        
        # Jacobi's formula: d(det A) = det(A) * tr(A^-1 * dA)
        # tr(A^-1 * dA) = sum_{i,j} (A^-1)_{ij} * (dA)_{ji}
        tr_inv_dA = np.einsum('ij, jik -> k', inv_real, self.dual)
        det_dual = det_real * tr_inv_dA
        
        return DualTensor(det_real, det_dual)

    def inv(self):
        """Inverse of a NxN DualTensor matrix."""
        inv_real = np.linalg.inv(self.real)
        
        # Matrix derivative rule: d(A^-1) = - A^-1 @ dA @ A^-1
        # Step 1: (A^-1)_{ij} * (dA)_{jlk} -> (inv_dA)_{ilk}
        inv_dA = np.einsum('ij, jlk -> ilk', inv_real, self.dual)
        
        # Step 2: (inv_dA)_{ijk} * (A^-1)_{jl} -> (inv_dual)_{ilk}
        # Contract spatial index 'j', preserving the AD dual index 'k'
        inv_dA_inv = np.einsum('ijk, jl -> ilk', inv_dA, inv_real)
        inv_dual = -inv_dA_inv
        
        return DualTensor(inv_real, inv_dual)

    def trace(self):
        """Trace of a NxN DualTensor matrix."""
        new_real = np.trace(self.real)
        # Trace over the spatial dimensions (axis 0 and 1), preserving the AD variables (axis 2)
        new_dual = np.trace(self.dual, axis1=0, axis2=1)
        return DualTensor(new_real, new_dual)

    # Elementary Operations
    def __add__(self, other):
        if isinstance(other, DualTensor):
            return DualTensor(self.real + other.real, self.dual + other.dual)
        return DualTensor(self.real + other, self.dual)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, DualTensor):
            self.real += other.real
            self.dual += other.dual
        else:
            self.real += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, DualTensor):
            return DualTensor(self.real - other.real, self.dual - other.dual)
        return DualTensor(self.real - other, self.dual)
    
    def __rsub__(self, other):
        return DualTensor(other - self.real, -self.dual)
    
    def __mul__(self, other):
        if isinstance(other, DualTensor):
            new_real = self.real * other.real
            # CORRECT Product rule: d(uv) = u'v + uv'
            new_dual = self.dual * other.real[..., None] + self.real[..., None] * other.dual
            return DualTensor(new_real, new_dual)

        if type(other) is np.ndarray:
            return DualTensor(self.real * other, self.dual * other[..., None])

        # Multiplying by a real scalar/array
        return DualTensor(self.real * other, self.dual * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, DualTensor):
            new_real = self.real / other.real
            new_dual = (self.dual * other.real[..., None] - self.real[..., None] * other.dual) / (other.real[..., None]**2)
            return DualTensor(new_real, new_dual)

        if type(other) is np.ndarray:
            return DualTensor(self.real / other, self.dual / other[..., None])
        
        return DualTensor(self.real / other, self.dual / other)
    
    def __rtruediv__(self, other):
        if type(other) is np.ndarray:
            new_real = other / self.real
            new_dual = -other[..., None] * self.dual / (self.real[..., None]**2)
            return DualTensor(new_real, new_dual)

        new_real = other / self.real
        new_dual = -other * self.dual / (self.real[..., None]**2)
        return DualTensor(new_real, new_dual)
    
    def __pow__(self, power):
        new_real = self.real ** power
        new_dual = power * (self.real[..., None] ** (power - 1)) * self.dual
        return DualTensor(new_real, new_dual)
        
    def __neg__(self):
        return DualTensor(-self.real, -self.dual)

    # --- Matrix Multiplication (@) ---
    def __matmul__(self, other):
        is_dual = isinstance(other, DualTensor)
        other_real = other.real if is_dual else np.asarray(other)
        
        # 1. Real matrix multiplication
        new_real = self.real @ other_real
        
        self_ndim = self.real.ndim
        other_ndim = other_real.ndim
        
        # 2. Dual product rule: dA @ B_real + A_real @ dB
        # Term 1: dA @ B_real
        if self_ndim == 2:
            term1 = np.tensordot(self.dual, other_real, axes=([1], [0]))
            if other_ndim == 2:
                term1 = np.swapaxes(term1, 1, 2)
        else:
            term1 = np.tensordot(self.dual, other_real, axes=([0], [0]))
            if other_ndim == 2:
                term1 = np.swapaxes(term1, 0, 1)
                
        # Term 2: A_real @ dB
        if is_dual:
            if self_ndim == 2:
                term2 = np.tensordot(self.real, other.dual, axes=([1], [0]))
            else:
                term2 = np.tensordot(self.real, other.dual, axes=([0], [0]))
        else:
            term2 = 0.0
            
        new_dual = term1 + term2
        return DualTensor(new_real, new_dual)

    def __rmatmul__(self, other):
        """
        Handles real_matrix @ self. 
        other: real array of shape (K,) or (M, K)
        self: DualTensor with real shape (K, C) and dual shape (K, C, V)
        """
        other_real = np.asarray(other)
        new_real = other_real @ self.real
        
        # tensordot(axes=1) automatically sums the last axis of other_real 
        # with the first axis of self.dual
        # If other_real is (K,), output is (C, V)
        # If other_real is (M, K), output is (M, C, V)
        new_dual = np.tensordot(other_real, self.dual, axes=1)
        
        return DualTensor(new_real, new_dual)

# ---------------------------------------------------------
# Elementary Math Wrappers
# ---------------------------------------------------------
def exp(x):
    if isinstance(x, DualTensor):
        val = np.exp(x.real)
        # val[..., None] broadcasts the real scalar/array across the dual dimension
        return DualTensor(val, val[..., None] * x.dual)
    return np.exp(x)

def log(x):
    if isinstance(x, DualTensor):
        return DualTensor(np.log(x.real), x.dual / x.real[..., None])
    return np.log(x)

def sin(x):
    if isinstance(x, DualTensor):
        return DualTensor(np.sin(x.real), np.cos(x.real)[..., None] * x.dual)
    return np.sin(x)

def cos(x):
    if isinstance(x, DualTensor):
        return DualTensor(np.cos(x.real), -np.sin(x.real)[..., None] * x.dual)
    return np.cos(x)

# The Jacobian

def jacobian(func, x_val):
    """
    Computes the real evaluation and Jacobian of a function using DualTensor.
    """
    x_real = np.asarray(x_val, dtype=float)
    N_vars = x_real.size
    
    # The derivative of x w.r.t itself is the identity matrix.
    # We reshape it so that the last dimension represents the independent variables.
    I = np.eye(N_vars)
    x_dual = I.reshape(x_real.shape + (N_vars,))
    
    x_tensor = DualTensor(x_real, x_dual)
    
    y_tensor = func(x_tensor)
    
    # Flatten the output duals into the standard Jacobian shape (N_outputs, N_inputs)
    J = y_tensor.dual.reshape((y_tensor.real.size, N_vars))
    return y_tensor.real, J