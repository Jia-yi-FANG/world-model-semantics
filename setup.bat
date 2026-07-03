@echo off
echo === Setup (Windows) ===

python -m venv venv_routeb
call venv_routeb\Scripts\activate

pip install -r requirements.txt

echo.
echo === Verify environment ===
python src/step1_test_env.py

echo.
echo Setup complete. Run: python src/step2_train_world_model.py
