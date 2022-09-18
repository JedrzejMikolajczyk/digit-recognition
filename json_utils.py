import json

#load json file
def load_from_json(json_file):
    with open(json_file, "r") as settings:
        models_dict = json.load(settings)
    return models_dict
  
def write_to_json(json_file, model_name, model_file, weights_file):
    with open(json_file, "r") as settings:
        models_dict = json.load(settings)
    models_dict[model_name] = {
        "model": model_file,
        "weights": weights_file}
    with open(json_file, "w") as settings:
        settings.write(json.dumps(models_dict))
