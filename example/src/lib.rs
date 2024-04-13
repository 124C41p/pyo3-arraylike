use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1};
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, PyResult, Python};
use pyo3_arraylike::PyArrayLike2;

#[pyfunction]
fn sum_of_rows<'py>(py: Python<'py>, ar: PyArrayLike2<'py, u32>) -> Bound<'py, PyArray1<u32>> {
    ar.view().sum_axis(Axis(0)).into_pyarray_bound(py)
}

#[pymodule]
fn example(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_of_rows, m)?)?;
    Ok(())
}
