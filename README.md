# PyO3-ArrayLike

This crate provides a single struct `PyArrayLike<T,D>` which can be used for extracting an array from any Python object which can be regarded as an array of type `T` and dimension `D` in a reasonable way.

## How it works

`PyArrayLike<T,D>` is basically an enum consisting of either a `PyReadonlyArray<T,D>` from the [rust-numpy](https://github.com/PyO3/rust-numpy) project (i.e. a readonly reference to an actual numpy array on the Python heap), or of an owned `Array<T,D>` from the [ndarray](https://github.com/rust-ndarray/ndarray) project. When calling `pyobject.extract::<PyArrayLike<T,D>>()` it immediately checks whether `pyobject` is already a numpy array of the right type and dimension by calling `pyobject.downcast::<PyReadonlyArray<T,D>>()` internally. Only if this fails, it will try to construct an owned array from `pyobject` by iterating over its elements and trying to convert them into the right type recursively.

## Example

Consider the following function. It takes anything which "looks like" an array of dimension two consisting of integers bewteen 0 and 4294967295. It computes the sum of the array's rows and returns a one dimensional numpy array.

```rust
#[pyfunction]
fn sum_of_rows<'py>(py: Python<'py>, ar: PyArrayLike2<'py, u32>) -> Bound<'py, PyArray1<u32>> {
    ar.view().sum_axis(Axis(0)).into_pyarray_bound(py)
}
```

You can call this function from Python in the following ways. The first two functions will succeed, the latter two will raise an error.


```python
def call1():
    """Succeeds by passing a reference of the input array to the rust component."""
    return sum_of_rows(np.array([[1,2,3],[4,5,6]], dtype='uint32'))

def call2():
    """Succeeds but allocates extra memory for rebuilding the input array inside the rust component."""
    return sum_of_rows([[1,2,3], iter([4,5,6]), np.array([7,8,9], dtype='int64')])

def call3():
    """Raises an error since the input array is expected to be two dimensional."""
    return sum_of_rows([1,2,3])

def call4():
    """Raises an error since the input array contains a value which cannot be safely casted to u32."""
    return sum_of_rows([[2**32,0,0], [0,0,0]])
```