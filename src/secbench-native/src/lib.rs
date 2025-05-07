// Copyright CEA (Commissariat à l'énergie atomique et aux
// énergies alternatives) (2017-2025)
//
// This software is governed by the CeCILL  license under French law and
// abiding by the rules of distribution of free software.  You can  use,
// modify and/ or redistribute the software under the terms of the CeCILL
// license as circulated by CEA, CNRS and INRIA at the following URL
// "http://www.cecill.info".
//
// As a counterpart to the access to the source code and  rights to copy,
// modify and redistribute granted by the license, users are provided only
// with a limited warranty  and the software's author,  the holder of the
// economic rights,  and the successive licensors  have only  limited
// liability.
//
// In this respect, the user's attention is drawn to the risks associated
// with loading,  using,  modifying and/or developing or reproducing the
// software by the user in light of its specific status of free software,
// that may mean  that it is complicated to manipulate,  and  that  also
// therefore means  that it is reserved for developers  and  experienced
// professionals having in-depth computer knowledge. Users are therefore
// encouraged to load and test the software's suitability as regards their
// requirements in conditions enabling the security of their systems and/or
// data to be ensured and,  more generally, to use and operate it in the
// same conditions as regards security.
//
// The fact that you are presently reading this means that you have had
// knowledge of the CeCILL license and that you accept its terms.

use pyo3::prelude::*;

#[pymodule]
fn secbench_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let register_submodule = |submodule: Bound<'_, PyModule>| -> PyResult<()> {
        m.add_submodule(&submodule)?;

        // HACK: Allows for importing only the submodule with `import secbench_native.something`
        m.py()
            .import_bound("sys")?
            .getattr("modules")?
            .set_item(format!("secbench_native.{}", submodule.name()?), submodule)?;
        Ok(())
    };

    // Meta stuff
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;

    // Actual modules
    #[cfg(feature = "secbench_processing")]
    register_submodule(secbench_processing::make_secbench_processing(m.py())?)?;

    Ok(())
}

#[pyfunction]
fn version() -> (u32, u32, u32) {
    let major = env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap();
    let minor = env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap();
    let patch = env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap();

    (major, minor, patch)
}

#[pyfunction]
fn features() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(feature = "secbench_processing")]
    features.push("processing".into());

    features
}