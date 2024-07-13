use numpy::ndarray::Axis;
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::{kmeans::kmeans, kmeans::KMeansConfig, quantize::ColorCruncher};
use numpy::{PyArray1, PyArray2, PyArray3};

#[pyfunction(name = "kmeans_3chan")]
#[doc = "Perform k-means clustering on a 3-channel dataset. Expects nx3 array of floats, returns nxk array of labels and kx3 array of centroids"]
fn py_kmeans_3chan(
    data: Vec<[f64; 3]>,
    k: usize,
) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray2<f32>>)> {
    let array = match numpy::ndarray::Array2::from_shape_vec((data.len(), 3), data) {
        Ok(array) => array,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to create array: {}",
                e
            )))
        }
    };
    let shape = array.shape();
    if shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 3-channel data, got {} channels",
            shape[1]
        )));
    }

    let array: Vec<[f32; 3]> = array
        .into_raw_vec()
        .into_iter()
        .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
        .collect();

    let config = KMeansConfig {
        k,
        ..Default::default()
    };

    let (clusters, centroids) = kmeans(&array, &config).unwrap();
    let centroids: Vec<Vec<f32>> = centroids
        .into_iter()
        .map(|c| vec![c[0], c[1], c[2]])
        .collect();

    Python::with_gil(|py| {
        let clusters = PyArray1::from_vec_bound(py, clusters);
        let centroids = PyArray2::from_vec2_bound(py, &centroids)
            .expect("Failed to convert centroids to PyArray2");
        Ok((clusters.to_owned().into(), centroids.to_owned().into()))
    })
}

#[pyfunction(name = "reduce_colorspace")]
#[doc = "Reduce the colorspace of a 3-channel dataset. Expects nxm x 3 array of bytes, returns nxm x k array of bytes"]
fn py_reduce_colorspace(
    data: PyReadonlyArray3<u8>,
    num_colors: i32,
    sample_rate: i32,
) -> PyResult<Py<PyArray3<u8>>> {
    let array = data.as_array();
    let shape = array.shape();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 3-channel data, got {} channels",
            shape[2]
        )));
    }

    let flattened: Vec<u8> = array
        .lanes(Axis(2))
        .into_iter()
        .map(|lane| {
            let slice = lane.as_slice().unwrap();
            [slice[0], slice[1], slice[2]]
        })
        .flatten()
        .collect();

    let quantizer = ColorCruncher::new(num_colors as usize, sample_rate as usize, 3);
    let data = quantizer.quantize_image(&flattened);

    let reshaped = match numpy::ndarray::Array3::from_shape_vec((shape[0], shape[1], 3), data) {
        Ok(reshaped) => reshaped,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to reshape array: {}",
                e
            )))
        }
    };

    Python::with_gil(|py| {
        Ok(PyArray3::from_owned_array_bound(py, reshaped)
            .to_owned()
            .into())
    })
}

#[pymodule]
fn colorcrunch(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_kmeans_3chan, m)?)?;
    m.add_function(wrap_pyfunction!(py_reduce_colorspace, m)?)?;
    Ok(())
}
