import json

def map_data(predictions, descriptions, texts, summaries):
    data = []
    
    for i in range(len(predictions['masks'])):
        obj_data = {
            "object_id": i,
            "description": descriptions[i],
            "text": texts[i],
            "summary": summaries[i]
        }
        data.append(obj_data)
    
    return data

def save_mapped_data(mapped_data, output_file='output/mapped_data.json'):
    with open(output_file, 'w') as f:
        json.dump(mapped_data, f, indent=2)
