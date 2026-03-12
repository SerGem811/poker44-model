#!/bin/bash
# setup.sh - Setup Poker44 validator environment
set -e

handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

success_msg() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

info_msg() {
  echo -e "\e[34m[INFO]\e[0m $1"
}

resolve_python() {
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done

  handle_error "Python 3.10+ is required."
}

check_python() {
  resolve_python
  info_msg "Using Python interpreter: $PYTHON_BIN"
  "$PYTHON_BIN" --version || handle_error "Failed to execute $PYTHON_BIN."
}

create_activate_venv() {
  VENV_DIR="validator_env"
  info_msg "Creating virtualenv in $VENV_DIR..."
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR" || handle_error "Failed to create virtualenv"
    success_msg "Virtualenv created."
  else
    info_msg "Virtualenv already exists. Skipping creation."
  fi

  info_msg "Activating virtualenv..."
  source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtualenv"
}

upgrade_pip() {
  info_msg "Upgrading pip and setuptools..."
  python -m pip install --upgrade pip setuptools || handle_error "Failed to upgrade pip/setuptools"
  success_msg "pip and setuptools upgraded."
}

install_python_reqs() {
  info_msg "Installing Python dependencies from requirements.txt..."
  [ -f "requirements.txt" ] || handle_error "requirements.txt not found"

  pip install -r requirements.txt || handle_error "Failed to install Python dependencies"
  success_msg "Dependencies installed."
}

install_modules() {
  info_msg "Installing current package in editable mode..."
  pip install -e . || handle_error "Failed to install current package"
  success_msg "Main package installed."
}

install_bittensor_cli() {
  info_msg "Installing bittensor-cli..."
  pip install bittensor-cli || handle_error "Failed to install bittensor-cli"
  success_msg "bittensor-cli installed."
}

show_completion_info() {
  echo
  success_msg "Poker44 validator environment configured!"
  echo
  echo -e "\e[33m[INFO]\e[0m Virtual environment: $(pwd)/validator_env"
  echo -e "\e[33m[INFO]\e[0m To activate: source validator_env/bin/activate"
  echo
  echo -e "\e[34m[NEXT STEPS]\e[0m"
  echo "1. Review scripts/validator/run/run_vali.sh and set wallet, hotkey, and private dataset path."
  echo "   source validator_env/bin/activate"
  echo "   ./scripts/validator/run/run_vali.sh"
}

main() {
  check_python
  create_activate_venv
  upgrade_pip
  install_python_reqs
  install_modules
  install_bittensor_cli
  show_completion_info
}

main "$@"
