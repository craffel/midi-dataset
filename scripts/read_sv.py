def get_sv_list(sv_file, delimiter='\t', skiplines=0, field_indices=None):
    '''
    Parses a delimiter-separated value file

    :parameters:
        - sv_file : str
            Path to the separated value file
        - skiplines : int
            Number of lines to skip at the beginning of the file
        - delimiter : str
            Delimiter used to separate values
        - field_indices : list of int or NoneType
            Desired field indices, if None then return all fields

    :returns:
        - sv_list : list of list
            One list per row of the sv file
    '''
    sv_list = []
    with open(sv_file, 'rb') as f:
        for line in f:
            fields = line.split(delimiter)
            if field_indices is None:
                sv_list.append(fields)
            else:
                sv_list.append([fields[n] for n in field_indices])
    # Remove first line - labels
    sv_list = sv_list[skiplines:]
    for n, line in enumerate(sv_list):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        sv_list[n] = line
    return sv_list
