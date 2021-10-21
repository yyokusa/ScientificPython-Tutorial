import numpy as np

a = np.arange(15).reshape(3, 5)
a.shape # (3, 5)
a.ndim # 2
a.dtype.name # 'int64'
a.itemsize # 8
a.size # 15
type(a) # <class 'numpy.ndarray'>
b = np.array((6, 7, 8))
type(b) # <class 'numpy.ndarray'>

# size of the array is known
np.zeros((3, 4)) # By default, dtype -> float64
np.ones((2, 3, 4), dtype=np.int16)
np.empty((2, 3)) # initial content is random
# creating sequence of numbers
np.arange(10, 30, 5) # array([10, 15, 20, 25])
np.arange(0, 2, 0.3) # array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])

from numpy import pi
np.linspace(0, 2, 9) # give me 9 numbers from 0 to 2
x = np.linspace(0, 2 * pi, 100) # useful to evaluate function at lots of points
f = np.sin(x)

# printing arrays
a = np.arange(6)                    # 1d array
print(a) # [0 1 2 3 4 5]
b = np.arange(12).reshape(4, 3)     # 2d array
print(b)
'''[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]'''
c = np.arange(24).reshape(2, 3, 4)  # 3d array
print(c)
'''[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]'''
np.set_printoptions(threshold=sys.maxsize)  # sys module should be imported

# Basic Operations
#   Arithmetic operators on arrays apply elementwise
a = np.array([20, 30, 40, 50])
b = np.arange(4) # array([0, 1, 2, 3])
c = a - b # array([20, 29, 38, 47])
b**2 # array([0, 1, 4, 9])
10 * np.sin(a) # array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
a < 35 # array([ True,  True, False, False])

# matrix product
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
A * B     # elementwise product
'''array([[2, 0],
       [0, 4]])'''
A @ B     # matrix product
'''array([[5, 4],
       [3, 4]])'''
A.dot(B)  # another matrix product
'''array([[5, 4],
       [3, 4]])'''

rg = np.random.default_rng(1)  # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
'''array([[3, 3, 3],
       [3, 3, 3]])'''
b += a
'''array([[3.51182162, 3.9504637 , 3.14415961],
       [3.94864945, 3.31183145, 3.42332645]])'''
a += b  # b is not automatically converted to integer type
# numpy.core._exceptions._UFuncOutputCastingError: 
# Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'

# Unary operators
a = rg.random((2,3))
a.sum()
a.min()
a.max()
b = np.arange(12).reshape(3, 4)
b.sum(axis=0) # sum of each column
b.min(axis=1) # min of each row
b.cumsum(axis=1) # cumulative sum along each row

# Indexing, Slicing and Iterating
# https://numpy.org/devdocs/user/quickstart.html#indexing-slicing-and-iterating

c = np.array([[[  0,  1,  2],  # a 3D array (two stacked 2D arrays)
               [ 10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])
c.shape # (2, 2, 3)
c[1, ...]  # same as c[1, :, :] or c[1]
'''array([[100, 101, 102],
       [110, 112, 113]])'''
c[..., 2]  # same as c[:, :, 2]
'''array([[  2,  13],
       [102, 113]])'''

# Shape manipulation
a = np.floor(10 * rg.random((3, 4)))
'''array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])'''
a.shape # (3, 4)
a.ravel() # returns flattened array
a.T # Transposed
a.T.shape
a.shape

'''
The reshape function returns its argument with a modified shape, 
whereas the ndarray.resize method modifies the array itself.

If a dimension is given as -1 in a reshaping operation, 
the other dimensions are automatically calculated.
'''
a = np.floor(10 * rg.random((2, 2)))
'''array([[9., 7.],
       [5., 2.]])'''
b = np.floor(10 * rg.random((2, 2)))
'''array([[1., 9.],
       [5., 1.]])'''
np.vstack((a, b))
'''array([[9., 7.],
       [5., 2.],
       [1., 9.],
       [5., 1.]])'''
np.hstack((a, b))
'''array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])'''
np.column_stack is np.hstack # False
np.row_stack is np.vstack # True

# Copies and Views
#   1 - No Copy at All
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object
# True

#   2 - View or Shallow Copy
c = a.view()
c is a # False
c.base is a            # c is a view of the data owned by a
# True
c.flags.owndata # False
c = c.reshape((2, 6))  # a's shape doesn't change
a.shape # (3, 4)
c[0, 4] = 1234         # a's data changes
a
'''array([[   0,    1,    2,    3],
    [1234,    5,    6,    7],
    [   8,    9,   10,   11]])'''
# Slicing an array returns a view of it:
s = a[:, 1:3]
s[:] = 10  # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
a
'''array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])'''

#   3 - Deep Copy
d = a.copy() # a new array object with new data is created
d is a # False
d.base is a  # d doesn't share anything with a
# False
d[0, 0] = 9999
a
'''array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])'''
a = np.arange(int(1e8))
b = a[:100].copy()
del a  # the memory of ``a`` can be released.

"""Functions and Methods Overview

Here is a list of some useful NumPy functions and methods names ordered in categories. See Routines for the full list.

Array Creation

    arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r_, zeros, zeros_like
Conversions

    ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat
Manipulations

    array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
Questions

    all, any, nonzero, where
Ordering

    argmax, argmin, argsort, max, min, ptp, searchsorted, sort
Operations

    choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum
Basic Statistics

    cov, mean, std, var
Basic Linear Algebra

    cross, dot, outer, linalg.svd, vdot"""

# Broadcasting rules
#   https://numpy.org/devdocs/user/basics.broadcasting.html#basics-broadcasting
# Advanced indexing and index tricks
a = np.arange(12)**2  # the first 12 square numbers
i = np.array([1, 1, 3, 8, 5])  # an array of indices
a[i]  # the elements of `a` at the positions `i`
# array([ 1,  1,  9, 64, 25])
j = np.array([[3, 4], [9, 7]])  # a bidimensional array of indices
a[j]  # the same shape as `j`
'''array([[ 9, 16],
        [81, 49]])'''

palette = np.array([[0, 0, 0],         # black
                    [255, 0, 0],       # red
                    [0, 255, 0],       # green
                    [0, 0, 255],       # blue
                    [255, 255, 255]])  # white
image = np.array([[0, 1, 2, 0],  # each value corresponds to a color in the palette
                  [0, 3, 4, 0]])
palette[image]  # the (2, 4, 3) color image
'''array([[[  0,   0,   0],
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0,   0]],

       [[  0,   0,   0],
        [  0,   0, 255],
        [255, 255, 255],
        [  0,   0,   0]]])'''

# Indexing with Boolean arrays
a = np.arange(12).reshape(3, 4)
b = a > 4
b  # `b` is a boolean with `a`'s shape
'''array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])'''
a[b]  # 1d array with the selected elements
# array([ 5,  6,  7,  8,  9, 10, 11])

# Histograms

'''The NumPy histogram function applied to an array returns a pair of vectors: 
the histogram of the array and a vector of the bin edges. 
Beware: matplotlib also has a function to build histograms (called hist, as in Matlab) 
that differs from the one in NumPy. The main difference is that pylab.hist 
plots the histogram automatically, while numpy.histogram only generates the data.'''

import numpy as np
rg = np.random.default_rng(1)
import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = rg.normal(mu, sigma, 10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=True)       # matplotlib version (plot)
# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5 * (bins[1:] + bins[:-1]), n)
