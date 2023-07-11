use indoc::indoc;
use ndarray::{Array, ArrayView, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use numpy::pyo3::Python;
use numpy::{
    ndarray::Dimension,
    pyo3::{
        exceptions::PyTypeError, sync::GILOnceCell, types::PyModule, FromPyObject, Py, PyAny,
        PyResult,
    },
    Element, IntoPyArray, PyArray, PyReadonlyArray,
};

pub struct PyArrayLike<'py, T, D>(ArrayLike<'py, T, D>)
where
    T: Element,
    D: Dimension;

#[derive(Debug)]
enum ArrayLike<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    PyRef(PyReadonlyArray<'py, T, D>),
    Owned(Array<T, D>),
}

impl<'py, T, D> PyArrayLike<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    pub fn into_owned_array(self) -> Array<T, D> {
        match self.0 {
            ArrayLike::PyRef(py_array) => py_array.to_owned_array(),
            ArrayLike::Owned(array) => array,
        }
    }

    pub fn into_pyarray(self, py: Python<'py>) -> PyReadonlyArray<T, D> {
        match self.0 {
            ArrayLike::PyRef(py_array) => py_array,
            ArrayLike::Owned(array) => array.into_pyarray(py).readonly(),
        }
    }

    pub fn view(&self) -> ArrayView<T, D> {
        match &self.0 {
            ArrayLike::PyRef(py_array) => py_array.as_array(),
            ArrayLike::Owned(array) => array.view(),
        }
    }
}

impl<'py, T, D> FromPyObject<'py> for PyArrayLike<'py, T, D>
where
    T: Element,
    D: Dimension,
    Vec<T>: FromPyObject<'py>,
{
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        if let Ok(array) = ob.downcast::<PyArray<T, D>>() {
            return Ok(PyArrayLike(ArrayLike::PyRef(array.readonly())));
        }

        let np_result = numpy_convert::<T, D>(ob)?;

        if let Ok(array) = np_result.extract::<(PyReadonlyArray<T, D>,)>() {
            return Ok(PyArrayLike(ArrayLike::PyRef(array.0)));
        }

        let (flattened_array, shape) = np_result.extract::<(Vec<T>, Vec<usize>)>()?;

        let result = Array::from_shape_vec(shape, flattened_array)
            .expect("Valid shape")
            .into_dimensionality()
            .map_err(|_| PyTypeError::new_err("Wrong dimensions"))?;

        Ok(PyArrayLike(ArrayLike::Owned(result)))
    }
}

fn numpy_convert<T, D>(array_like: &PyAny) -> PyResult<&PyAny>
where
    T: Element,
    D: Dimension,
{
    static NP_CONVERTER_CACHE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    let py = array_like.py();

    let fun = NP_CONVERTER_CACHE.get_or_try_init(py, || {
        let fun = PyModule::from_code(
            py,
            indoc! {"
                import numpy as np
                def convert(ar, dt):
                    ar = np.asarray(ar)
                    try:
                        return ar.astype(dt, casting = 'safe', copy = False),
                    except:
                        return ar.flatten(), ar.shape
            "},
            "",
            "",
        )?
        .getattr("convert")
        .expect("Attribute exists");
        PyResult::Ok(Py::from(fun))
    });

    fun?.as_ref(py).call((array_like, T::get_dtype(py)), None)
}

pub type PyArrayLike0<'py, T> = PyArrayLike<'py, T, Ix0>;
pub type PyArrayLike1<'py, T> = PyArrayLike<'py, T, Ix1>;
pub type PyArrayLike2<'py, T> = PyArrayLike<'py, T, Ix2>;
pub type PyArrayLike3<'py, T> = PyArrayLike<'py, T, Ix3>;
pub type PyArrayLike4<'py, T> = PyArrayLike<'py, T, Ix4>;
pub type PyArrayLike5<'py, T> = PyArrayLike<'py, T, Ix5>;
pub type PyArrayLike6<'py, T> = PyArrayLike<'py, T, Ix6>;
pub type PyArrayLikeDyn<'py, T> = PyArrayLike<'py, T, IxDyn>;
