#!/usr/bin/env bash

set -ex

cd $(dirname $0)

# if [[ -f "~/souper/third_party/z3-install/lib/libz3.so" ]]; then
#     export LD_LIBRARY_PATH="~/souper/third_party/z3-install/lib"
# fi

cargo build --example brahma --release --all-features

echo "###############################################################################" >> run.stdout.log
echo "# Running with '$@'"                                                             >> run.stdout.log
echo "###############################################################################" >> run.stdout.log
export RUST_LOG=synth_loop_free_prog=debug
cargo run --example brahma --release --all-features -- $@           2>> run.stderr.log >> run.stdout.log
