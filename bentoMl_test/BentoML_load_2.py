import bentoml

# Load BentoML model
model_ref = bentoml.picklable_model.get("energy_consumption_forecast_xgb:latest")

# Convert to a runner
runner = model_ref.to_runner()

# Initialize the runner for local testing
runner.init_local()

# Example input (raw values, no scaling needed)
raw_input = [8156881, 8042235, 8483870, 8078961, 8157491, 8205930,
             8056152, 8146142, 8120391, 8205399, 8222668, 8082651]

# Run prediction directly with raw input (scaling is handled inside the model)
prediction = runner.run(raw_input)

print(f"Predicted Consumption: {prediction}")
