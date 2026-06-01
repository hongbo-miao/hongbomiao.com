#!/usr/bin/env bash
set -e

if [ "$(uname)" != "Darwin" ]; then
  echo "This installer targets macOS only."
  exit 1
fi

airband_release_tag="v5.2.0"
soapy_airspy_release_tag="soapy-airspy-0.2.0"
install_prefix="${HOME}/.local"
job_count="$(sysctl -n hw.ncpu)"

# Clone and build in a throwaway temp dir; only the installed binaries persist.
build_root="$(mktemp -d)"
trap 'rm -rf "${build_root}"' EXIT

brew install lame libshout libconfig fftw librtlsdr airspy soapysdr cmake pkg-config git

# SoapyAirspy is not in Homebrew (the tap formula is broken), so build it from source.
# CMAKE_POLICY_VERSION_MINIMUM works around its old cmake_minimum_required;
# installing into the Homebrew prefix lets the SoapySDR runtime auto-load the module.
brew_prefix="$(brew --prefix)"
soapy_airspy_dir="${build_root}/SoapyAirspy"
git clone --depth 1 --branch "${soapy_airspy_release_tag}" \
  https://github.com/pothosware/SoapyAirspy.git "${soapy_airspy_dir}"
cmake -S "${soapy_airspy_dir}" -B "${soapy_airspy_dir}/build" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX="${brew_prefix}"
cmake --build "${soapy_airspy_dir}/build" -j "${job_count}"
cmake --install "${soapy_airspy_dir}/build"

airband_dir="${build_root}/RTLSDR-Airband"
git clone --depth 1 --branch "${airband_release_tag}" \
  https://github.com/rtl-airband/RTLSDR-Airband.git "${airband_dir}"
cmake -S "${airband_dir}" -B "${airband_dir}/build" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DPLATFORM=generic -DNFM=ON -DRTLSDR=ON -DSOAPYSDR=ON
cmake --build "${airband_dir}/build" -j "${job_count}"
cmake --install "${airband_dir}/build"

echo "Installed rtl_airband to ${install_prefix}/bin"
if ! command -v rtl_airband >/dev/null; then
  # shellcheck disable=SC2016  # intentional literal for the user to copy
  echo 'Add to PATH: export PATH="$HOME/.local/bin:$PATH"'
fi
