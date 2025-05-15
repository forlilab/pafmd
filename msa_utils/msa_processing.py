import itertools 

def format_sto(sto_path: str) -> bool:
    """ Format sto file such that GC RF line is the second to last line (i.e preceding //)
        This is to work with AF function remove_empty_columns_from_stockholm_msa
        Returns if the .sto file is valid
    Args:
        sto_path: path to .sto file (e.g mgnify.sto or uniref90_hits.sto)
    """  


    with open(sto_path, 'r') as f:
        stockholm_msa = f.read()
    
    stockholm_msa = stockholm_msa.splitlines()

    for i,row in enumerate(stockholm_msa):
        if '#=GC RF' in row:
            gc_rf_idx = i 
            break 

    gc_rf_line = stockholm_msa[gc_rf_idx]
    last_line = stockholm_msa[-1]

    del stockholm_msa[gc_rf_idx]
    del stockholm_msa[-1]

    stockholm_msa.append(gc_rf_line)
    stockholm_msa.append(last_line)
      
    stockholm_msa_str = '\n'.join(stockholm_msa)
 
    with open(sto_path, 'w') as f:
        f.write(stockholm_msa_str)



def remove_empty_columns_from_stockholm_msa(stockholm_msa: str) -> str:
  """Removes empty columns (dashes-only) from a Stockholm MSA."""
  """AF2 function assumes that GC RF occurs AFTER alignment""" 
  processed_lines = {}
  unprocessed_lines = {}
  for i, line in enumerate(stockholm_msa.splitlines()):
    if line.startswith('#=GC RF'):
      reference_annotation_i = i 
      reference_annotation_line = line
      # Reached the end of this chunk of the alignment. Process chunk.
      _, _, first_alignment = line.rpartition(' ')
      mask = []
      for j in range(len(first_alignment)):
        for _, unprocessed_line in unprocessed_lines.items():
          prefix, _, alignment = unprocessed_line.rpartition(' ')
          if alignment[j] != '-':
            mask.append(True)
            break
        else:  # Every row contained a hyphen - empty column.
          mask.append(False)
      # Add reference annotation for processing with mask.
      unprocessed_lines[reference_annotation_i] = reference_annotation_line

      if not any(mask):  # All columns were empty. Output empty lines for chunk.
        for line_index in unprocessed_lines:
          processed_lines[line_index] = ''
      else:
        for line_index, unprocessed_line in unprocessed_lines.items():
          prefix, _, alignment = unprocessed_line.rpartition(' ')
          masked_alignment = ''.join(itertools.compress(alignment, mask))
          processed_lines[line_index] = f'{prefix} {masked_alignment}'

      # Clear raw_alignments.
      unprocessed_lines = {}
    elif line.strip() and not line.startswith(('#', '//')):
      unprocessed_lines[i] = line
    else:
      processed_lines[i] = line

  ###added code###
  try: 
    return '\n'.join((processed_lines[i] for i in range(len(processed_lines))))
  except KeyError:
    return '-1'  


def check_sto_is_valid(stockholm_msa: str) -> bool:
    out = remove_empty_columns_from_stockholm_msa(stockholm_msa)
    if out == '-1':
        return False
    else:
        return True


