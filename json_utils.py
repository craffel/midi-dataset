import json


def create_js(data, field_names, output_file):
    '''
    Create a json file from a list of lists with the provided field names

    :parameters:
        - data : list of list
            Each entry is a list of field values
        - field_names : list of str
            Names of the fields in data, len(field_names) = len(data[n])
        - output_file : str
            Path to where the json file should be written
    '''
    output_json = [dict(zip(field_names, row)) for row in data]
    with open(output_file, 'wb') as f:
        json.dump(output_json, f, indent=4)
