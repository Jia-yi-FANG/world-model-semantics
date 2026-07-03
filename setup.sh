#!/bin/bash
echo "=== Setup (Mac/Linux) ==="

python3 -m venv venv_routeb
source venv_routeb/bin/activate

pip install -r requirements.txt

echo ""
echo "=== Verify environment ==="
python src/step1_test_env.py

echo ""
echo "Setup complete. Run: python src/step2_train_world_model.py"
