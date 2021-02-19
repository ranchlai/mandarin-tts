""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols

from ipdb import set_trace
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
unk_idx = len(symbols)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
   # set_trace()
    text = text.replace('}','')
    text = text.replace('{','')
    
    sequence = _arpabet_to_sequence(_clean_text(text, cleaner_names))

  

    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
  #  set_trace()
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s +' '
        else:
            result += 'unk' + ' '
    return result.replace('}{', ' ')[:-1]


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
       #set_trace()
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    seq = []
    for s in symbols:
        if _should_keep_symbol(s):
            seq.append(_symbol_to_id[s])
        else:
            seq.append(unk_idx)
            
    return seq 
    #return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'
