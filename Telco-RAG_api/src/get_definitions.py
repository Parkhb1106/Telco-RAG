from docx import Document
import os

def read_docx(file_path):
    """Reads a .docx file and categorizes its content into terms and abbreviations."""
    doc = Document(file_path)
    
    processing_terms = False
    processing_abbreviations = False
    start = 0 
    terms_definitions = {}
    abbreviations_definitions = {}
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if "References" in text:
            start += 1
        if start >= 2:
            if "Terms and definitions" in text:
                processing_terms = True
                processing_abbreviations = False
                
            elif "Abbreviations" in text:
                processing_abbreviations = True
                processing_terms = False
            else:
                if processing_terms and ':' in text:
                    term, definition = text.split(':', 1)
                    terms_definitions[term.strip()] = definition.strip().rstrip('.')
                elif processing_abbreviations and '\t' in text:
                    abbreviation, definition = text.split('\t', 1)
                    if len(abbreviation) > 1:
                        abbreviations_definitions[abbreviation.strip()] = definition.strip()
                
    return terms_definitions, abbreviations_definitions

def preprocess(text, lowercase=True):
    """Converts text to lowercase and removes punctuation."""
    if lowercase:
        text = text.lower()
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in punctuations:
        text = text.replace(char, '')
    return text

def find_and_filter_terms(terms_dict, sentence):
    """Finds terms in the given sentence, case-insensitively, and filters out shorter overlapping terms."""
    lowercase_sentence = preprocess(sentence, lowercase=True)
    
    matched_terms = {term: terms_dict[term] for term in terms_dict if preprocess(term) in lowercase_sentence}
    
    final_terms = {}
    for term in matched_terms:
        if not any(term in other and term != other for other in matched_terms):
            final_terms[term] = matched_terms[term]
            
    return final_terms

def find_and_filter_abbreviations(abbreviations_dict, sentence):
    """Finds abbreviations in the given sentence, case-sensitively, and filters out shorter overlapping abbreviations."""
    processed_sentence = preprocess(sentence, lowercase=False)  
    words = processed_sentence.split() 
    
    matched_abbreviations = {word: abbreviations_dict[word] for word in words if word in abbreviations_dict}
    
    final_abbreviations = {}
    sorted_abbrs = sorted(matched_abbreviations, key=len, reverse=True)
    for abbr in sorted_abbrs:
        if not any(abbr in other and abbr != other for other in sorted_abbrs):
            final_abbreviations[abbr] = matched_abbreviations[abbr]
    
    return final_abbreviations

def find_terms_and_abbreviations_in_sentence(terms_dict, abbreviations_dict, sentence):
    """Finds and filters terms and abbreviations in the given sentence.
       Abbreviations are matched case-sensitively, terms case-insensitively, and longer terms are prioritized."""
    matched_terms = find_and_filter_terms(terms_dict, sentence)
    matched_abbreviations = find_and_filter_abbreviations(abbreviations_dict, sentence)

    formatted_terms = [f"{term}: {definition}" for term, definition in matched_terms.items()]
    formatted_abbreviations = [f"{abbr}: {definition}" for abbr, definition in matched_abbreviations.items()]

    return formatted_terms, formatted_abbreviations

def get_def(sentence):
    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    file_path = os.path.join(resources_dir, "3GPP_vocabulary.docx")
    terms_definitions, abbreviations_definitions = read_docx(file_path)
    formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, sentence)
    defined = []
    for term in formatted_terms:
        defined.append(term[:3])
    for abbreviation in formatted_abbreviations:
        defined.append(abbreviation[:3])

def define_TA_question(sentence):
    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    file_path = os.path.join(resources_dir, "3GPP_vocabulary.docx")
    terms_definitions, abbreviations_definitions = read_docx(file_path)
    # add custom term, abbreviation definitions if needed
    #"""
    custom_terms_definitions = {
        'Numerology': 'A term referring to the configuration of subcarrier spacing and cyclic prefix length, which allows 5G NR to support diverse services and frequency bands flexibly.',
        'Msg1': 'Colloquial term for the Random Access Preamble transmitted by the UE during the initial access (RACH) procedure.',
        'Msg2': 'Colloquial term for the Random Access Response (RAR) sent by the gNB in response to Msg1.',
        'Msg3': 'Colloquial term for the RRC Connection Request (or similar initial uplink transmission) sent by the UE on the PUSCH resources allocated by Msg2.',
        'Msg4': 'Colloquial term for the Contention Resolution message sent by the gNB to finalize the random access procedure.',
        'P1 Procedure': 'Beam management procedure used for initial beam selection, involving a wide beam sweep by the gNB and UE.',
        'P2 Procedure': 'Beam management procedure used for beam refinement of the transmitter (gNB) side, typically narrowing down the beam width.',
        'P3 Procedure': 'Beam management procedure used for beam refinement of the receiver (UE) side, where the UE adjusts its Rx beam while the gNB beam is fixed.',
        'K0': 'A parameter in DCI that defines the slot delay between the downlink DCI reception and the corresponding PDSCH (data) reception.',
        'K1': 'A parameter in DCI that defines the slot delay between PDSCH reception and the corresponding HARQ-ACK feedback transmission on PUCCH.',
        'K2': 'A parameter in DCI that defines the slot delay between the uplink DCI reception and the corresponding PUSCH (data) transmission.',
        'Initial Active BWP': 'The default Bandwidth Part used by the UE during the initial access phase before receiving dedicated BWP configurations.',
        'Search Space': 'A set of candidate control channel elements (CCEs) where the UE attempts to blindly decode PDCCH candidates.',
    }
    custom_abbreviations_definitions = {
        'CORESET': 'Control Resource Set',
        'BWP': 'Bandwidth Part',
        'SSB': 'Synchronization Signal / PBCH Block',
        'SCS': 'Subcarrier Spacing',
        'RMSI': 'Remaining Minimum System Information',
        'OSI': 'Other System Information',
        'SLIV': 'Start and Length Indicator Value',
        'RIV': 'Resource Indication Value',
        'TCI': 'Transmission Configuration Indicator',
        'PTRS': 'Phase Tracking Reference Signal',
        'CCE': 'Control Channel Element',
        'REG': 'Resource Element Group',
        'RA-RNTI': 'Random Access Radio Network Temporary Identifier',
        'TC-RNTI': 'Temporary Cell Radio Network Temporary Identifier',
        'P-RNTI': 'Paging Radio Network Temporary Identifier',
        'SI-RNTI': 'System Information Radio Network Temporary Identifier',
        'CS-RNTI': 'Configured Scheduling Radio Network Temporary Identifier',
        'MCS-C-RNTI': 'Modulation and Coding Scheme Cell Radio Network Temporary Identifier',
        'FR1': 'Frequency Range 1',
        'FR2': 'Frequency Range 2',
        'CBG': 'Code Block Group',
    }
    for k, v in custom_terms_definitions.items():
        terms_definitions.setdefault(k, v)
    for k, v in custom_abbreviations_definitions.items():
        abbreviations_definitions.setdefault(k, v)
    #"""
    formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, sentence)
    terms = '\n'.join(formatted_terms)
    abbreviations = '\n'.join(formatted_abbreviations)
    question = f"""{sentence}\n
Terms and Definitions:\n
{terms}\n

Abbreviations:\n
{abbreviations}\n
"""
    dictionary = f"""Terms and Definitions:\n
{terms}\n

Abbreviations:\n
{abbreviations}\n
"""
    return question, dictionary
