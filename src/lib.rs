//! This crate provides a single struct `PyArrayLike<T,D>` which can be used for extracting an array from any Python object which can be regarded as an array of type `T` and dimension `D` in a reasonable way.

#![deny(missing_docs, missing_debug_implementations)]

#[cfg(test)]
mod test;

use ndarray::{Array, ArrayView, Axis, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use numpy::{
    ndarray::Dimension,
    pyo3::{
        exceptions::PyValueError, types::PyAnyMethods, Bound, Borrowed, PyAny, PyErr, PyResult, FromPyObject, Python
    },
    Element, IntoPyArray, PyArray, PyArrayMethods, PyReadonlyArray,
};
use std::fmt::Debug;

/// To be used for extracting an array from any Python object which can be regarded as an array of type `T` and dimension `D` in a reasonable way.
#[derive(Debug)]
pub struct PyArrayLike<'py, T, D>(ArrayLike<'py, T, D>)
where
    T: Element,
    D: Dimension;

enum ArrayLike<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    PyRef(PyReadonlyArray<'py, T, D>),
    Owned(Array<T, D>, Python<'py>),
}

impl<'py, T, D> Debug for ArrayLike<'py, T, D>
where
    T: Element + Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PyRef(py_array) => f.debug_tuple("PyRef").field(py_array).finish(),
            Self::Owned(array, _) => f.debug_tuple("Owned").field(array).finish(),
        }
    }
}

impl<'py, T, D> PyArrayLike<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    /// Consumes `self` and moves its data into an owned array.
    pub fn into_owned_array(self) -> Array<T, D> {
        match self.0 {
            ArrayLike::PyRef(py_array) => py_array.to_owned_array(),
            ArrayLike::Owned(array, _) => array,
        }
    }

    /// Consumes `self` and moves its data into a numpy array.
    pub fn into_pyarray(self) -> PyReadonlyArray<'py, T, D> {
        match self.0 {
            ArrayLike::PyRef(py_array) => py_array,
            ArrayLike::Owned(array, py) => array.into_pyarray(py).readonly(),
        }
    }

    /// Return a read-only view of the array.
    pub fn view<'s>(&'s self) -> ArrayView<'s, T, D> {
        match &self.0 {
            ArrayLike::PyRef(py_array) => py_array.as_array(),
            ArrayLike::Owned(array, _) => array.view(),
        }
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self.0 {
            ArrayLike::PyRef(py_array) => py_array.as_slice().ok(),
            ArrayLike::Owned(array, _) => array.as_slice(),
        }
    }

    /// Return the array’s dimension
    pub fn dim(&self) -> D::Pattern {
        match &self.0 {
            ArrayLike::PyRef(py_array) => py_array.dims().into_pattern(),
            ArrayLike::Owned(array, _) => array.dim(),
        }
    }
}

impl<'py, T, D> From<PyArrayLike<'py, T, D>> for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn from(value: PyArrayLike<'py, T, D>) -> Self {
        value.into_pyarray()
    }
}

impl<T, D> From<PyArrayLike<'_, T, D>> for Array<T, D>
where
    T: Element,
    D: Dimension,
{
    fn from(value: PyArrayLike<T, D>) -> Self {
        value.into_owned_array()
    }
}

impl<'py, T, D> PyArrayLike<'py, T, D>
where
    T: Clone + Element + 'static + for<'a> FromPyObject<'a, 'py>,
    D: Dimension + 'static,
{
    fn from_python(ob: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(array) = ob.cast::<PyArray<T, D>>() {
            return Some(PyArrayLike(ArrayLike::PyRef(array.readonly())));
        }

        if matches!(D::NDIM, None | Some(0)) {
            if let Ok(value) = ob.extract::<T>() {
                let res = Array::from_elem((), value).into_dimensionality().ok()?;
                return Some(PyArrayLike(ArrayLike::Owned(res, ob.py())));
            }
        }

        if matches!(D::NDIM, None | Some(1)) {
            if let Ok(array) = ob.extract::<Vec<T>>() {
                let res = Array::from_vec(array).into_dimensionality().ok()?;
                return Some(PyArrayLike(ArrayLike::Owned(res, ob.py())));
            }
        }

        let sub_arrays = ob
            .try_iter()
            .ok()?
            .map(|item| {
                item.ok()
                    .and_then(|ob| <PyArrayLike<T, D::Smaller>>::from_python(&ob))
            })
            .collect::<Option<Vec<_>>>()?;
        let sub_array_views = sub_arrays.iter().map(|x| x.view()).collect::<Vec<_>>();
        let array = ndarray::stack(Axis(0), &sub_array_views)
            .ok()?
            .into_dimensionality()
            .ok()?;
        Some(PyArrayLike(ArrayLike::Owned(array, ob.py())))
    }
}

impl<'py, T, D> FromPyObject<'_, 'py> for PyArrayLike<'py, T, D>
where
    T:  Clone + Element + 'static + for<'a> FromPyObject<'a, 'py>,
    D: Dimension + 'static,
{
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        Self::from_python(&ob).ok_or_else(|| {
            let dtype = T::get_dtype(ob.py());
            let err_text = match D::NDIM {
                Some(dim) => format!("Expected an array like of dimension {} containing elements which can be safely casted to {}.", dim, dtype),
                None => format!("Expected an array like of arbitrary dimension containing elements which can be safely casted to {}.", dtype)
            };
            PyValueError::new_err(err_text)})
    }
}

/// Zero-dimensional array like.
pub type PyArrayLike0<'py, T> = PyArrayLike<'py, T, Ix0>;
/// One-dimensional array like.
pub type PyArrayLike1<'py, T> = PyArrayLike<'py, T, Ix1>;
/// Two-dimensional array like.
pub type PyArrayLike2<'py, T> = PyArrayLike<'py, T, Ix2>;
/// Three-dimensional array like.
pub type PyArrayLike3<'py, T> = PyArrayLike<'py, T, Ix3>;
/// Four-dimensional array like.
pub type PyArrayLike4<'py, T> = PyArrayLike<'py, T, Ix4>;
/// Five-dimensional array like.
pub type PyArrayLike5<'py, T> = PyArrayLike<'py, T, Ix5>;
/// Six-dimensional array like.
pub type PyArrayLike6<'py, T> = PyArrayLike<'py, T, Ix6>;
/// Array like of any dimension.
pub type PyArrayLikeDyn<'py, T> = PyArrayLike<'py, T, IxDyn>;
