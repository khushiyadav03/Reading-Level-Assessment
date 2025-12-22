import textstat

def extract_features(text):
    
    words = text.split()
    word_count = len(words)

    sentence_count = max(
        1,
        text.count('.') + text.count('!') + text.count('?')
    )

    avg_sentence_length = word_count / sentence_count

    return {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length
    }
