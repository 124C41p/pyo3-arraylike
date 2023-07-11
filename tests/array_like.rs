use ndarray::{array, Array0};
use numpy::get_array_module;
use pyo3::{
    types::{IntoPyDict, PyDict},
    Python,
};
use pyo3_arraylike::{PyArrayLike0, PyArrayLike1, PyArrayLike2, PyArrayLikeDyn};

fn get_np_locals(py: Python) -> &PyDict {
    [("np", get_array_module(py).unwrap())].into_py_dict(py)
}

#[test]
fn extract_reference() {
    Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let py_array = py
            .eval(
                "np.array([[1,2],[3,4]], dtype='float64')",
                Some(locals),
                None,
            )
            .unwrap();
        let extracted_array = py_array.extract::<PyArrayLike2<f64>>().unwrap();

        assert_eq!(
            array![[1_f64, 2_f64], [3_f64, 4_f64]],
            extracted_array.into_owned_array()
        );
    });
}

#[test]
fn convert_array_on_extract() {
    Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let py_array = py
            .eval("np.array([[1,2],[3,4]], dtype='int')", Some(locals), None)
            .unwrap();
        let extracted_array = py_array.extract::<PyArrayLike2<f64>>().unwrap();

        assert_eq!(
            array![[1_f64, 2_f64], [3_f64, 4_f64]],
            extracted_array.into_owned_array()
        );
    });
}

#[test]
fn convert_list_on_extract() {
    Python::with_gil(|py| {
        let py_list = py.eval("[[1,2],[3,4]]", None, None).unwrap();
        let extracted_array = py_list.extract::<PyArrayLike2<i32>>().unwrap();

        assert_eq!(array![[1, 2], [3, 4]], extracted_array.into_owned_array());
    });
}

#[test]
fn convert_array_in_list_on_extract() {
    Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let py_array = py
            .eval(
                "[np.array([1, 2], dtype='int32'), [3, 4]]",
                Some(locals),
                None,
            )
            .unwrap();
        let extracted_array = py_array.extract::<PyArrayLike2<i32>>().unwrap();

        assert_eq!(array![[1, 2], [3, 4]], extracted_array.into_owned_array());
    });
}

#[test]
fn convert_list_on_extract_dyn() {
    Python::with_gil(|py| {
        let py_list = py
            .eval("[[[1,2],[3,4]],[[5,6],[7,8]]]", None, None)
            .unwrap();
        let extracted_array = py_list.extract::<PyArrayLikeDyn<i32>>().unwrap();

        assert_eq!(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            extracted_array.into_owned_array()
        );
    });
}

#[test]
fn convert_1d_list_on_extract() {
    Python::with_gil(|py| {
        let py_list = py.eval("[1,2,3,4]", None, None).unwrap();
        let extracted_array_1d = py_list.extract::<PyArrayLike1<u32>>().unwrap();
        let extracted_array_dyn = py_list.extract::<PyArrayLikeDyn<f64>>().unwrap();

        assert_eq!(array![1, 2, 3, 4], extracted_array_1d.into_owned_array());
        assert_eq!(
            array![1_f64, 2_f64, 3_f64, 4_f64].into_dyn(),
            extracted_array_dyn.into_owned_array()
        );
    });
}

#[test]
fn unsafe_cast_shall_fail() {
    Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let py_list = py
            .eval(
                "np.array([1.1,2.2,3.3,4.4], dtype='float64')",
                Some(locals),
                None,
            )
            .unwrap();
        let extracted_array = py_list.extract::<PyArrayLike1<i32>>();

        assert!(extracted_array.is_err());
    });
}

#[test]
fn extract_0d_array() {
    Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let array0 = py
            .eval("np.array(1, dtype='int64')", Some(locals), None)
            .unwrap();
        let num = py.eval("42", None, None).unwrap();

        let extraction1 = array0.extract::<PyArrayLike0<i32>>().unwrap();
        let extraction2 = num.extract::<PyArrayLike0<i32>>().unwrap();
        let extraction3 = num.extract::<PyArrayLikeDyn<usize>>().unwrap();

        assert_eq!(extraction1.into_owned_array(), Array0::from_elem((), 1));
        assert_eq!(extraction2.into_owned_array(), Array0::from_elem((), 42));
        assert_eq!(
            extraction3.into_owned_array().into_dyn(),
            Array0::from_elem((), 42).into_dyn()
        );
    });
}