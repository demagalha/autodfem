import numpy as np

class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        return Dual(self.real + other, self.dual)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        return Dual(self.real - other, self.dual)
    
    def __rsub__(self, other):
        return Dual(other - self.real, -self.dual)
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            new_real = self.real * other.real
            new_dual = self.real * other.dual + self.dual * other.real
            return Dual(new_real, new_dual)
        return Dual(self.real * other, self.dual * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Dual):
            new_real = self.real / other.real
            new_dual = (self.dual * other.real - self.real * other.dual) / (other.real**2)
            return Dual(new_real, new_dual)
        return Dual(self.real / other, self.dual / other)
    
    def __rtruediv__(self, other):
        new_real = other / self.real
        new_dual = -other * self.dual / (self.real**2)
        return Dual(new_real, new_dual)
    
    def __pow__(self, power):
        new_real = self.real ** power
        new_dual = power * (self.real ** (power - 1)) * self.dual
        return Dual(new_real, new_dual)
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)
    

# ---------------------------------------------------------
# Elementary Math Wrappers
# ---------------------------------------------------------
def exp(x):
    if isinstance(x, Dual):
        val = np.exp(x.real)
        return Dual(val, val * x.dual)
    return np.exp(x)

def log(x):
    if isinstance(x, Dual):
        return Dual(np.log(x.real), x.dual / x.real)
    return np.log(x)

def sin(x):
    if isinstance(x, Dual):
        return Dual(np.sin(x.real), np.cos(x.real) * x.dual)
    return np.sin(x)

def cos(x):
    if isinstance(x, Dual):
        return Dual(np.cos(x.real), -np.sin(x.real) * x.dual)
    return np.cos(x)


# The Jacobian

def jacobian(func, x_val):
    """
    Computes the real evaluation and Jacobian of a function using Dual numbers.
    """
    n = len(x_val)
    I = np.eye(n)
    x_dual = [Dual(x_val[i], I[i]) for i in range(n)]
    y_dual = func(x_dual)
    
    y_real = np.array([yi.real for yi in y_dual])
    J = np.array([yi.dual for yi in y_dual])
    return y_real, J