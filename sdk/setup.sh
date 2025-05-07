###
# Copyright CEA (Commissariat à l'énergie atomique et aux
# énergies alternatives) (2017-2025)
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
###

set -eu -o pipefail

function __log_info() {
    echo -e "[\e[32minfo\e[0m] $1"
}

function __log_error() {
    echo -e "[\e[91merror\e[0m] $1"
}

function __setup_or_update_secbench() {
    local sdk_root
    local secbench_src
    local conda_root
    local env_name

    env_name="$2"
    sdk_root="$1"
    conda_root="${sdk_root}/mambaforge"

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        __log_error "You are attempting to run this script within an Anaconda environment, which can cause errors in the installation. Please exit the Anaconda envirnoment and retry."
        exit 1
    fi

    __log_info "SECBENCH SDK installation path is \"${sdk_root}\"" && mkdir -p "${sdk_root}"

    secbench_src="$(git rev-parse --show-toplevel || true)"
    if [[ -d "${secbench_src}/sdk" ]]; then
        __log_info "source repository \"%s\"" "${secbench_src}"
    else
        exit 1
    fi

    if [[ ! -O "${conda_root}/etc/profile.d/mamba.sh" ]]; then
        local miniforge
        miniforge="Miniforge3-$(uname)-$(uname -m).sh"
        miniforge_out="${USER:-undefuser}_${miniforge}"
        (cd /tmp && wget -O "${miniforge_out}" "https://github.com/conda-forge/miniforge/releases/latest/download/${miniforge}" && chmod +x "${miniforge_out}" && "./${miniforge_out}" -b -f -p "${conda_root}" -s) || true
    fi

    # Create anaconda environment
    # shellcheck source=/dev/null
    source "${conda_root}/etc/profile.d/conda.sh"
    conda_env_available="$(conda env list | grep -i "^${env_name} " || true)"
    if [[ -n "${conda_env_available}" ]]; then
        __log_info "\"${env_name}\" anaconda environment already installed, updating packages"
        mamba update -n "${env_name}" --all -y
    else
        __log_info "creating anaconda environment \"${env_name}\""
        conda create -n "${env_name}" --override-channels -c conda-forge -y python=3.11 mamba
    fi

    # Packages in anaconda or conda-forge repositories
    conda_pkgs=(
        asciitree
        click
        cython
        flake8
        furo
        lttbc
        h5py
        ipython
        jupyter
        jupytext
        matplotlib
        maturin
        mypy
        myst-nb
        numba  # Comment if this causes troubles
        numpy
        pint
        pyftdi
        pylibftdi
        pyserial
        pytest
        python-dateutil
        python-vxi11
        pyudev
        pandas
        pyusb
        pyvisa
        ruff
        sphinx
        scikit-learn
        scipy
        setuptools
        tabulate
        tensorflow
        tqdm
    )
    __log_info "installing anaconda packages (from anaconda repository or conda-forge). This can take a few minutes to complete..."
    mamba install -n "${env_name}" -y "${conda_pkgs[@]}"
    # Packages not in anaconda or conda-forge channels.
    #
    # **IMPORTANT**: we install those packages with --no-deps to avoid changing
    # versions of packages installed by conda.
    extra_pkgs=()

    __log_info "applying patches to specific packages."
    mamba run -n "${env_name}" python "${secbench_src}/sdk/patch_libudev.py"

    # __log_info "installing extra dependencies through pip"
    # mamba run -n "${env_name}" python -m pip install --no-deps "${extra_pkgs[@]}"

    secbench_dir="./src"
    secbench_dev_pkgs=(
        "$secbench_dir/secbench-api"
        "$secbench_dir/secbench-storage"
        "$secbench_dir/secbench-picoscope"
        "$secbench_dir/secbench-native"
        "$secbench_dir/secbench-processing")

    __log_info "installing secbench packages in developpment mode through pip"
    # NOTE: we need to install them one at a time, otherwise the '-e' option
    # is not applied.
    for pkg in "${secbench_dev_pkgs[@]}"; do
        __log_info "installing ${pkg}"
        mamba run -n "${env_name}" python -m pip install --no-color --progress-bar off --no-deps -e "${pkg}"
    done

    # Generate the activate script in the SDK installation directory
    cp "${secbench_src}/sdk/activate.sh.in" "${sdk_root}/activate"
    sed -i "s+@@SECBENCH_SDK_ROOT@@+\"${sdk_root}\"+g" "${sdk_root}/activate"
    sed -i "s+@@SECBENCH_SDK_ENV_NAME@@+\"${env_name}\"+g" "${sdk_root}/activate"

    __log_info "installation is successfull! Please run 'source ${sdk_root}/activate' to activate the developpment environment"
}

function show_usage() {
    echo -e "Usage: $1 [-h|--help] [-p|--prefix <prefix>]\n"
    echo "Options:"
    echo "  -h, --help      Display this help message"
    echo "  -p, --prefix    Specify the installation path of the SDK (defaults to ~/tools/secbench_sdk, or a path pointed by the SECBENCH_SDK_ROOT environment variable)"
    echo "  -e, --env-name  Name of the miniforge environment environment (default: secbench-dev)"
}

function setup_or_update_secbench() {
    local install_prefix="${SECBENCH_SDK_ROOT:-${HOME}/tools/secbench_sdk}"
    local program_name="$0"
    local env_name="secbench-dev"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        key="$1"

        case $key in
            -h|--help)
                show_usage "${program_name}"
                exit 0
                ;;
            -p|--prefix)
                install_prefix="$2"
                shift
                ;;
            -e|--env-name)
                env_name="$2"
                shift
                ;;
            *)
                __log_error "invalid argument $1"
                exit 1
                ;;
        esac
        shift
    done
    __setup_or_update_secbench "${install_prefix}" "${env_name}"
}

setup_or_update_secbench "$@"
